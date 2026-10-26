//===- HipPasses.cpp ------------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Define a pass plugin that runs a collection of HIP passes.
//
// (c) 2021 Parmance for Argonne National Laboratory and
// (c) 2022 Pekka Jääskeläinen / Intel
// (c) 2023 chipStar developers
// (c) 2024 Henry Linjamäki / Intel
//===----------------------------------------------------------------------===//

#include "HipAbort.h"
#include "HipCleanup.h"
#include "HipDynMem.h"
#include "HipStripDebugInfo.h"
#include "HipStripUsedIntrinsics.h"
#include "HipWarps.h"
#include "HipPrintf.h"
#include "HipGlobalVariables.h"
#include "HipTextureLowering.h"
#include "HipEmitLoweredNames.h"
#include "HipKernelArgSpiller.h"
#include "HipLowerZeroLengthArrays.h"
#include "HipSanityChecks.h"
#include "LLVMSPIRV.h"
#include "HipLowerSwitch.h"
#include "HipLowerMemset.h"
#include "HipLowerHintIntrinsics.h"
#include "HipLowerFPAtomicMinMax.h"
#include "HipLowerRoundIntrinsics.h"
#include "HipLowerSubwordAtomics.h"
#include "HipLowerVolatileAccesses.h"
#include "HipIGBADetector.h"
#include "HipFunctionPointerAS.h"
#include "HipPromoteInts.h"
#include "HipLowerOverflowIntrinsics.h"
#include "HipLowerPointerVectors.h"
#include "HipSpirvFunctionReorderPass.h"
#include "HipVerify.h"
#include "HipCanonicalizeGEP.h"
#include "HipCoalesceDuplicatePhiPreds.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "PassPluginCompat.h"
#include "chipStarConfig.hh"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

using namespace llvm;

// A predicate for internalize pass. Returning true means preserve GV.
//
// Internalizes all non-kernel functions so unused ones get removed by DCE
// pass, and the Itanium vtable family (_ZTV vtable, _ZTT VTT, _ZTC construction
// vtable). Clang emits those for every polymorphic class that appears in device
// code, and an explicit instantiation gives them weak_odr linkage, which
// GlobalDCE alone must keep. A device module is self-contained, so nothing can
// link against them and the unreferenced ones can go; left in, a dead table
// whose virtual base offsets are inttoptr constants aborts llvm-spirv
// (CHIP-SPV/chipStar#1382).
static bool preserveDuringInternalize(const GlobalValue &GV) {
  if (isa<GlobalVariable>(GV)) {
    StringRef Name = GV.getName();
    return !(Name.starts_with("_ZTV") || Name.starts_with("_ZTT") ||
             Name.starts_with("_ZTC"));
  }
  const auto *F = dyn_cast<Function>(&GV);
  return !(F && F->getCallingConv() == CallingConv::SPIR_FUNC);
}

// Strip compiler-generated optnone+noinline pairs (debug builds) but
// preserve user-annotated __attribute__((noinline)) so it propagates
// to SPIR-V FunctionControl=DontInline.  When both optnone and
// noinline are present, they came from the compiler; noinline alone
// means the user asked for it explicitly.
class RemoveNoInlineOptNoneAttrsPass
    : public HipRequiredPassInfoMixin<RemoveNoInlineOptNoneAttrsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    for (auto &F : M) {
      if (F.hasFnAttribute(Attribute::OptimizeNone)) {
        F.removeFnAttr(Attribute::NoInline);
        F.removeFnAttr(Attribute::OptimizeNone);
      }
    }
    return PreservedAnalyses::none();
  }
  static bool isRequired() { return true; }
};

// LLVM commit 15a1769631ff0b2b3e830b03e51ae5f54f08a0ab introduces
// 'opencl.ocl.version' module metadata into device code. This triggers an
// assertion in SPIRV-LLVM Translator if the HIP device code and linked device
// bitcode has mixed OpenCL version metadata. The commit in question inserts
// OpenCL version 0.0 in HIP compilation mode (in CUDA mode it is 2.0).
//
// This pass works around the issue until some fix is introduced in Clang. The
// issue is fixed by setting OpenCL version to the same as bitcode library
// (2.0).
class HipFixOpenCLMDPass : public HipRequiredPassInfoMixin<HipFixOpenCLMDPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    constexpr auto OCLVersionMDName = "opencl.ocl.version";
    if (auto *OCLVersionMD = M.getNamedMetadata(OCLVersionMDName)) {
      auto &Ctx = M.getContext();
      auto *Int32Ty = IntegerType::get(Ctx, 32);
      M.eraseNamedMetadata(OCLVersionMD);
      Metadata *OCLVerElts[] = {
          ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 2)),
          ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 0))};
      OCLVersionMD = M.getOrInsertNamedMetadata(OCLVersionMDName);
      OCLVersionMD->addOperand(MDNode::get(Ctx, OCLVerElts));
    }
    // Altering OpenCL metadata probably does not invalidate any analyses.
    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

// WORKAROUND(CHIP-SPV/chipStar#1691, llvm/llvm-project#198078): clang hoists a
// local aggregate initializer taking __shared__ addresses into a constant
// global, where IGC reads them as null and PoCL aborts. Remove when a clang fix
// that initializes such locals in the function lands.
class HipSharedAddrLocalInitPass
    : public HipRequiredPassInfoMixin<HipSharedAddrLocalInitPass> {
  static bool refersToShared(const Value *V) {
    if (const auto *GV = dyn_cast<GlobalValue>(V))
      return GV->getAddressSpace() == SPIRV_WORKGROUP_AS;
    const auto *C = dyn_cast<Constant>(V);
    return C && any_of(C->operands(),
                       [](const Use &Op) { return refersToShared(Op.get()); });
  }

  // Stores C to Ptr, element by element where it refers to shared memory.
  static void storeInit(Constant *C, AllocaInst *Ptr,
                        SmallVectorImpl<Value *> &Idx, IRBuilder<> &B) {
    if (refersToShared(C) &&
        (isa<ConstantArray>(C) || isa<ConstantStruct>(C))) {
      for (unsigned I = 0; I < C->getNumOperands(); ++I) {
        Idx.push_back(B.getInt32(I));
        storeInit(C->getAggregateElement(I), Ptr, Idx, B);
        Idx.pop_back();
      }
      return;
    }
    Type *Ty = Ptr->getAllocatedType();
    uint64_t Off =
        Ptr->getModule()->getDataLayout().getIndexedOffsetInType(Ty, Idx);
    B.CreateAlignedStore(C, B.CreateInBoundsGEP(Ty, Ptr, Idx),
                         commonAlignment(Ptr->getAlign(), Off));
  }

public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    const DataLayout &DL = M.getDataLayout();
    SmallSetVector<GlobalVariable *, 4> Inits;
    for (Function &F : M)
      for (Instruction &I : make_early_inc_range(instructions(F))) {
        auto *Copy = dyn_cast<MemCpyInst>(&I);
        if (!Copy)
          continue;
        Value *Src = Copy->getRawSource();
        APInt Off(DL.getIndexTypeSizeInBits(Src->getType()), 0);
        auto *GV = dyn_cast<GlobalVariable>(
            Src->stripAndAccumulateConstantOffsets(DL, Off, true));
        if (!GV || !GV->isConstant() || !GV->hasDefinitiveInitializer() ||
            !refersToShared(GV->getInitializer()))
          continue;
        // Copy from a function-local instance of the initializer instead.
        IRBuilder<> B(&F.getEntryBlock(),
                      F.getEntryBlock().getFirstInsertionPt());
        AllocaInst *Init = B.CreateAlloca(GV->getValueType());
        B.SetInsertPoint(Copy);
        SmallVector<Value *, 4> Idx{B.getInt32(0)};
        storeInit(GV->getInitializer(), Init, Idx, B);
        Value *NewSrc = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), Init,
                                                     Off.getZExtValue());
        B.CreateMemCpy(Copy->getRawDest(), Copy->getDestAlign(), NewSrc,
                       commonAlignment(Init->getAlign(), Off.getZExtValue()),
                       Copy->getLength(), Copy->isVolatile());
        Copy->eraseFromParent();
        Inits.insert(GV);
      }
    for (GlobalVariable *GV : Inits) {
      GV->removeDeadConstantUsers();
      if (GV->use_empty() && GV->hasLocalLinkage())
        GV->eraseFromParent();
    }
    return Inits.empty() ? PreservedAnalyses::all() : PreservedAnalyses::none();
  }
  static bool isRequired() { return true; }
};

#ifdef CHIP_LLVM_USE_INTERGRATED_SPIRV
// WORKAROUND(CHIP-SPV/chipStar#1654, llvm/llvm-project#206404): the in-tree
// backend puts ContractionOff on every kernel unless opencl.enable.FP_CONTRACT
// is present, llvm-spirv only on kernels reaching an op that forbids
// contraction. Remove when the backend's default decides from the operations.
class HipFPContractPass : public HipRequiredPassInfoMixin<HipFPContractPass> {
  // What makes llvm-spirv disable contraction for the enclosing function.
  static bool forbidsContraction(const Instruction &I) {
    if (auto *B = dyn_cast<BinaryOperator>(&I))
      return (B->getOpcode() == Instruction::FAdd ||
              B->getOpcode() == Instruction::FSub) &&
             !B->hasAllowContract();
    auto *CI = dyn_cast<CallInst>(&I);
    if (!CI || isa<IntrinsicInst>(CI))
      return false;
    const Function *Callee = CI->getCalledFunction();
    if (!Callee)
      return true;
    if (!Callee->isDeclaration())
      return false;
    // Builtins are named printf, __spirv_*, or mangled as _Z<len><name>.
    StringRef Name = Callee->getName();
    if (Name.size() > 2 && Name.starts_with("_Z") && isDigit(Name[2]))
      Name = Name.drop_front(2).ltrim("0123456789");
    else if (Name != "printf" && !Name.starts_with("__spirv_"))
      return true;
    // Any other __ name is a chipStar runtime helper, not a builtin.
    return Name.starts_with("__") && !Name.starts_with("__spirv_");
  }

public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    for (const Function &F : M)
      for (const Instruction &I : instructions(F))
        if (forbidsContraction(I))
          return PreservedAnalyses::all();
    M.getOrInsertNamedMetadata("opencl.enable.FP_CONTRACT");
    return PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

// WORKAROUND(CHIP-SPV/chipStar#1958, llvm/llvm-project (not yet reported)): the
// in-tree backend emits llvm.memmove as OpCopyMemorySized, which consumers read
// as memcpy. Remove when the SPIR-V backend lowers memmove itself.
class HipLowerMemmovePass
    : public HipRequiredPassInfoMixin<HipLowerMemmovePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    SmallVector<MemMoveInst *, 4> MemMoves;
    for (Instruction &I : instructions(F))
      if (auto *MemMove = dyn_cast<MemMoveInst>(&I))
        MemMoves.push_back(MemMove);
    const TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);
    bool Changed = false;
    for (MemMoveInst *MemMove : MemMoves) {
      unsigned DstAS = MemMove->getDestAddressSpace();
      unsigned SrcAS = MemMove->getSourceAddressSpace();
      // Compare mixed address spaces as generic; constant memory cannot overlap.
      if (DstAS != SrcAS && DstAS != 2 && SrcAS != 2) {
        IRBuilder<> B(MemMove);
        Type *Generic = B.getPtrTy(4);
        auto *Copy = cast<MemMoveInst>(B.CreateMemMove(
            B.CreateAddrSpaceCast(MemMove->getRawDest(), Generic),
            MemMove->getDestAlign(),
            B.CreateAddrSpaceCast(MemMove->getRawSource(), Generic),
            MemMove->getSourceAlign(), MemMove->getLength(),
            MemMove->isVolatile()));
        MemMove->eraseFromParent();
        MemMove = Copy;
        Changed = true;
      }
      if (expandMemMoveAsLoop(MemMove, TTI)) {
        MemMove->eraseFromParent();
        Changed = true;
      }
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};
#endif

// WORKAROUND(CHIP-SPV/chipStar#1891, KhronosGroup/SPIRV-LLVM-Translator#3918): translator lowers sitofp i1 true to +1.0. Remove when the fix is in the translator release branch chipStar builds.
// WORKAROUND(CHIP-SPV/chipStar#1891, llvm/llvm-project#209232): the in-tree
// backend does the same. Remove when the fix is in the LLVM release chipStar
// builds. Delete the pass only once CanarySitofpI1Translator and
// CanarySitofpI1Backend both fire for every supported LLVM version.
// Rewrites sitofp i1 %c into select %c, -1.0, 0.0.
class HipLowerSitofpI1Pass
    : public HipRequiredPassInfoMixin<HipLowerSitofpI1Pass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    bool Changed = false;
    for (Instruction &I : make_early_inc_range(instructions(F))) {
      auto *Cast = dyn_cast<SIToFPInst>(&I);
      if (!Cast || !Cast->getSrcTy()->isIntOrIntVectorTy(1))
        continue;
      Type *Ty = Cast->getType();
      auto *Sel = SelectInst::Create(Cast->getOperand(0),
                                     ConstantFP::get(Ty, -1.0),
                                     ConstantFP::get(Ty, 0.0), "",
                                     Cast->getIterator());
      Sel->takeName(Cast);
      Cast->replaceAllUsesWith(Sel);
      Cast->eraseFromParent();
      Changed = true;
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

// WORKAROUND(CHIP-SPV/chipStar#1896, intel/intel-graphics-compiler (not yet reported)): IGC aliases int views of a float in SIMD32 shuffles. Remove when IGC fixes emitSimdShuffle.
// Rewrites sub_group_shuffle{,_xor}(bitcast float %f to i32) as the float
// shuffle of %f.
class HipFloatShufflePass
    : public HipRequiredPassInfoMixin<HipFloatShufflePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    SmallVector<CallInst *, 8> Calls;
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I); CI && CI->getCalledFunction())
        if (StringRef N = CI->getCalledFunction()->getName();
            N == "_Z17sub_group_shuffleij" || N == "_Z17sub_group_shufflejj" ||
            N == "_Z21sub_group_shuffle_xorij" ||
            N == "_Z21sub_group_shuffle_xorjj")
          if (auto *Cast = dyn_cast<BitCastInst>(CI->getArgOperand(0));
              Cast && Cast->getSrcTy()->isFloatTy())
            Calls.push_back(CI);

    for (CallInst *CI : Calls) {
      auto *Cast = cast<BitCastInst>(CI->getArgOperand(0));
      Function *Callee = CI->getCalledFunction();
      Type *FloatTy = Cast->getSrcTy();
      std::string FloatName = (Callee->getName().drop_back(2) + "fj").str();
      Function *FloatFn = F.getParent()->getFunction(FloatName);
      if (!FloatFn) {
        FloatFn = Function::Create(
            FunctionType::get(
                FloatTy, {FloatTy, CI->getArgOperand(1)->getType()}, false),
            Callee->getLinkage(), FloatName, F.getParent());
        FloatFn->copyAttributesFrom(Callee);
      }
      auto *FloatCI = CallInst::Create(
          FloatFn, {Cast->getOperand(0), CI->getArgOperand(1)}, "",
          CI->getIterator());
      FloatCI->setCallingConv(CI->getCallingConv());
      FloatCI->setAttributes(CI->getAttributes());
      FloatCI->takeName(CI);
      for (User *U : make_early_inc_range(CI->users()))
        if (auto *Back = dyn_cast<BitCastInst>(U);
            Back && Back->getDestTy() == FloatTy) {
          Back->replaceAllUsesWith(FloatCI);
          Back->eraseFromParent();
        }
      if (!CI->use_empty())
        CI->replaceAllUsesWith(
            new BitCastInst(FloatCI, CI->getType(), "", CI->getIterator()));
      CI->eraseFromParent();
      if (Cast->use_empty())
        Cast->eraseFromParent();
    }
    return Calls.empty() ? PreservedAnalyses::all() : PreservedAnalyses::none();
  }
  static bool isRequired() { return true; }
};

// WORKAROUND(CHIP-SPV/chipStar#1978, intel/intel-graphics-compiler: no upstream issue yet): IGC lowers a constant-offset sub_group_shuffle to an a0 indirect gather. Remove when IGC lowers it like a shuffle down.
// Rewrites sub_group_shuffle(v, I) to __chip_sg_shift_<T>(v, I, D) when I is
// proven to be the lane, or the lane plus D, and under a run-time test at the
// function entry otherwise. The rtdevlib defines both per device.
class HipShuffleShiftPass
    : public HipRequiredPassInfoMixin<HipShuffleShiftPass> {
  // An OpenCL work-item query, e.g. _Z12get_local_idj.
  static bool isWorkItemQuery(const Value *V, bool SizeOnly = false) {
    auto *CI = dyn_cast<CallInst>(V);
    Function *F = CI ? CI->getCalledFunction() : nullptr;
    if (!F || !F->isDeclaration() || CI->hasOperandBundles())
      return false;
    StringRef N = F->getName();
    if (!N.consume_front("_Z"))
      return false;
    N = N.drop_while([](char C) { return isDigit(C); });
    if (SizeOnly)
      return N.starts_with("get_") && N.contains("size");
    return N.starts_with("get_global_") || N.starts_with("get_local_") ||
           N.starts_with("get_group_id") || N.starts_with("get_num_groups") ||
           N.starts_with("get_enqueued_local_size") ||
           N.starts_with("get_sub_group_") ||
           N.starts_with("get_max_sub_group_size") ||
           N.starts_with("get_num_sub_groups");
  }

  // True if V can be recomputed at the function entry: it reads only
  // work-item queries, constants and arguments, and cannot divide by zero.
  // Sets D to the first constant added to or subtracted from a value.
  static bool isHoistable(Value *V, int64_t &D, SmallPtrSet<Value *, 16> &Seen,
                          int Depth = 0) {
    if (isa<Constant>(V) || isa<Argument>(V) || Seen.contains(V))
      return true;
    auto *I = dyn_cast<Instruction>(V);
    if (!I || Depth > 16)
      return false;
    Seen.insert(V);
    if (isa<CallInst>(I))
      return isWorkItemQuery(I) &&
             all_of(cast<CallInst>(I)->args(), [](Value *A) {
               return isa<ConstantInt>(A);
             });
    if (I->isIntDivRem() || I->isShift()) {
      // No division by zero, no shift past the width.
      Value *Op = I->getOperand(1);
      auto *C = dyn_cast<ConstantInt>(Op);
      if (isa<ZExtInst, SExtInst>(Op))
        Op = cast<CastInst>(Op)->getOperand(0);
      bool Signed = I->getOpcode() == Instruction::SDiv ||
                    I->getOpcode() == Instruction::SRem;
      if (I->isShift()
              ? !C || C->getValue().uge(I->getType()->getIntegerBitWidth())
              : !(C && (Signed ? C->getValue().isStrictlyPositive()
                               : !C->isZero())) &&
                    !isWorkItemQuery(Op, true))
        return false;
    } else if (!isa<BinaryOperator, ZExtInst, SExtInst, TruncInst, ICmpInst,
                   SelectInst>(I)) {
      return false;
    }
    if (auto *C = dyn_cast<ConstantInt>(I->getOperand(I->getNumOperands() - 1));
        C && !D && !C->isZero() && C->getValue().getSignificantBits() <= 32 &&
        (I->getOpcode() == Instruction::Add ||
         I->getOpcode() == Instruction::Sub))
      D = I->getOpcode() == Instruction::Add ? C->getSExtValue()
                                             : -C->getSExtValue();
    return all_of(I->operands(), [&](Value *Op) {
      return isHoistable(Op, D, Seen, Depth + 1);
    });
  }

  // Copies a hoistable V to before At, each instruction once.
  static Value *hoist(Value *V, Instruction *At,
                      DenseMap<Value *, Value *> &Copies) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return V;
    if (Value *Copy = Copies.lookup(I))
      return Copy;
    Instruction *New = I->clone();
    for (Use &Op : New->operands())
      if (!isa<Function>(Op))
        Op = hoist(Op, At, Copies);
    New->insertBefore(At->getIterator());
    return Copies[I] = New;
  }

  // V for sub-group local id L and max sub-group size M, if V depends on
  // nothing else.
  using EvalCache = DenseMap<Value *, std::optional<APInt>>;
  static std::optional<APInt> eval(Value *V, unsigned L, unsigned M,
                                   EvalCache &Cache, int Depth = 0) {
    if (auto *C = dyn_cast<ConstantInt>(V))
      return C->getValue();
    if (auto It = Cache.find(V); It != Cache.end())
      return It->second;
    auto R = evalUncached(V, L, M, Cache, Depth);
    Cache[V] = R;
    return R;
  }
  static std::optional<APInt> evalUncached(Value *V, unsigned L, unsigned M,
                                           EvalCache &Cache, int Depth) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || Depth > 32 || !I->getType()->isIntegerTy())
      return std::nullopt;
    unsigned Bits = I->getType()->getIntegerBitWidth();
    if (auto *CI = dyn_cast<CallInst>(I)) {
      Function *F = CI->getCalledFunction();
      if (!F || CI->hasOperandBundles())
        return std::nullopt;
      if (F->getName() == "_Z22get_sub_group_local_idv")
        return APInt(Bits, L);
      if (F->getName() == "_Z22get_max_sub_group_sizev")
        return APInt(Bits, M);
      return std::nullopt;
    }
    SmallVector<APInt, 3> Ops;
    for (Value *Op : I->operands()) {
      auto A = eval(Op, L, M, Cache, Depth + 1);
      if (!A)
        return std::nullopt;
      Ops.push_back(*A);
    }
    if (isa<SelectInst>(I))
      return Ops[0].isOne() ? Ops[1] : Ops[2];
    if (auto *Cmp = dyn_cast<ICmpInst>(I))
      return APInt(1, ICmpInst::compare(Ops[0], Ops[1], Cmp->getPredicate()));
    switch (I->getOpcode()) {
    case Instruction::ZExt:
      return Ops[0].zext(Bits);
    case Instruction::SExt:
      return Ops[0].sext(Bits);
    case Instruction::Trunc:
      return Ops[0].trunc(Bits);
    case Instruction::Add:
      return Ops[0] + Ops[1];
    case Instruction::Sub:
      return Ops[0] - Ops[1];
    case Instruction::Mul:
      return Ops[0] * Ops[1];
    case Instruction::And:
      return Ops[0] & Ops[1];
    case Instruction::Or:
      return Ops[0] | Ops[1];
    case Instruction::Xor:
      return Ops[0] ^ Ops[1];
    case Instruction::Shl:
      return Ops[1].uge(Bits) ? std::nullopt : std::optional(Ops[0].shl(Ops[1]));
    case Instruction::LShr:
      return Ops[1].uge(Bits) ? std::nullopt : std::optional(Ops[0].lshr(Ops[1]));
    case Instruction::AShr:
      return Ops[1].uge(Bits) ? std::nullopt : std::optional(Ops[0].ashr(Ops[1]));
    case Instruction::UDiv:
      return Ops[1].isZero() ? std::nullopt : std::optional(Ops[0].udiv(Ops[1]));
    case Instruction::URem:
      return Ops[1].isZero() ? std::nullopt : std::optional(Ops[0].urem(Ops[1]));
    case Instruction::SDiv:
      return Ops[1].isZero() ? std::nullopt : std::optional(Ops[0].sdiv(Ops[1]));
    case Instruction::SRem:
      return Ops[1].isZero() ? std::nullopt : std::optional(Ops[0].srem(Ops[1]));
    default:
      return std::nullopt;
    }
  }

  // D if, for every max sub-group size the native module accepts, each lane L
  // reads lane L, L + D, or L + D wrapped by the sub-group size.
  static std::optional<int> provenShift(Value *Idx) {
    std::optional<int> D;
    for (unsigned M : {32u, 8u, 16u, 64u})
      for (unsigned L = 0; L < M; L++) {
        EvalCache Cache;
        auto I = eval(Idx, L, M, Cache);
        if (!I || I->getBitWidth() != 32 || I->uge(4 * M))
          return std::nullopt;
        int Diff = (int)I->getZExtValue() - (int)L;
        if (!D && Diff)
          D = std::abs(Diff - (int)M) < std::abs(Diff) ? Diff - (int)M : Diff;
        if (Diff && Diff != *D && Diff != *D - (int)M && Diff != *D + (int)M)
          return std::nullopt;
      }
    return D;
  }

  static CallInst *call(Module &M, StringRef Name, Type *Ret,
                        ArrayRef<Value *> Args, BasicBlock::iterator At) {
    SmallVector<Type *, 3> Tys;
    for (Value *A : Args)
      Tys.push_back(A->getType());
    FunctionCallee F =
        M.getOrInsertFunction(Name, FunctionType::get(Ret, Tys, false));
    cast<Function>(F.getCallee())->setCallingConv(CallingConv::SPIR_FUNC);
    cast<Function>(F.getCallee())->addFnAttr(Attribute::Convergent);
    auto *CI = CallInst::Create(F, Args, "", At);
    CI->setCallingConv(CallingConv::SPIR_FUNC);
    return CI;
  }

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    SmallVector<CallInst *, 8> Sites;
    for (Instruction &I : instructions(F)) {
      // The calls this adds would mix uncontrolled with controlled convergence.
      if (auto *CB = dyn_cast<CallBase>(&I);
          CB && CB->getOperandBundle(LLVMContext::OB_convergencectrl))
        return PreservedAnalyses::all();
      if (auto *CI = dyn_cast<CallInst>(&I); CI && CI->getCalledFunction() &&
                                             !CI->hasOperandBundles() &&
                                             !CI->isMustTailCall())
        if (StringRef N = CI->getCalledFunction()->getName();
            N.size() == 23 && N.starts_with("_Z17sub_group_shuffle") &&
            StringRef("ijlmf").contains(N[21]))
          Sites.push_back(CI);
    }

    Module &M = *F.getParent();
    Type *I32 = Type::getInt32Ty(M.getContext());
    Instruction *Entry = &*F.getEntryBlock().getFirstInsertionPt();
    bool Changed = false;
    for (CallInst *CI : Sites) {
      Type *Ty = CI->getType();
      std::string Name = "__chip_sg_shift_";
      Name += CI->getCalledFunction()->getName()[21];
      // The index is a shift by construction: no test needed.
      if (auto PD = provenShift(CI->getArgOperand(1))) {
        CallInst *Shift =
            call(M, Name, Ty,
                 {CI->getArgOperand(0), CI->getArgOperand(1),
                  ConstantInt::get(I32, *PD, true)},
                 CI->getIterator());
        CI->replaceAllUsesWith(Shift);
        CI->eraseFromParent();
        Changed = true;
        continue;
      }
      int64_t D = 0;
      SmallPtrSet<Value *, 16> Seen;
      if (!isHoistable(CI->getArgOperand(1), D, Seen) || !D || D != (int32_t)D)
        continue;
      DenseMap<Value *, Value *> Copies;
      // Frozen: lanes that never reach the shuffle may compute poison here.
      Value *Idx = new FreezeInst(hoist(CI->getArgOperand(1), Entry, Copies),
                                  "", Entry->getIterator());
      Value *DV = ConstantInt::get(I32, D, true);
      auto *Ok = new ICmpInst(
          Entry->getIterator(), ICmpInst::ICMP_NE,
          call(M, "__chip_sg_shift_ok", I32, {Idx, DV}, Entry->getIterator()),
          ConstantInt::get(I32, 0));
      Instruction *Then, *Else;
      SplitBlockAndInsertIfThenElse(Ok, CI->getIterator(), &Then, &Else);
      BasicBlock *Tail = CI->getParent();
      CI->moveBefore(Else->getIterator());
      CallInst *Fast =
          call(M, Name, Ty, {CI->getArgOperand(0), CI->getArgOperand(1), DV},
               Then->getIterator());
      PHINode *Phi = PHINode::Create(Ty, 2, "", Tail->begin());
      CI->replaceAllUsesWith(Phi);
      Phi->addIncoming(Fast, Then->getParent());
      Phi->addIncoming(CI, Else->getParent());
      Changed = true;
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

// WORKAROUND(CHIP-SPV/chipStar#1703, no upstream report): the SPIR-V backend
// indexes an initializer's byte offset into the global's own type. Remove
// when it offsets by bytes.
static bool retypeToBytes(GlobalVariable &GV) {
  Type *I8 = Type::getInt8Ty(GV.getContext());
  if (auto *Ty = dyn_cast<ArrayType>(GV.getValueType());
      Ty && Ty->getElementType() == I8)
    return true;
  if (!GV.hasLocalLinkage() || !GV.hasDefinitiveInitializer())
    return false;
  const DataLayout &DL = GV.getParent()->getDataLayout();
  Constant *Init = GV.getInitializer();
  auto *Ty = ArrayType::get(I8, DL.getTypeAllocSize(GV.getValueType()));
  Constant *Bytes = ConstantAggregateZero::get(Ty);
  if (!Init->isNullValue()) {
    // The most constituents one OpConstantComposite can hold.
    if (Ty->getNumElements() > 65532)
      return false;
    SmallVector<uint8_t, 64> Data;
    for (uint64_t I = 0; I < Ty->getNumElements(); ++I) {
      // Null for a byte of a pointer or of undef.
      auto *B = dyn_cast_or_null<ConstantInt>(
          ConstantFoldLoadFromConst(Init, I8, APInt(64, I), DL));
      if (!B)
        return false;
      Data.push_back(B->getZExtValue());
    }
    Bytes = ConstantDataArray::get(GV.getContext(), Data);
  }
  if (!GV.getAlign())
    GV.setAlignment(DL.getPreferredAlign(&GV));
  GV.replaceInitializer(Bytes);
  return true;
}

// WORKAROUND(CHIP-SPV/chipStar#1693, CHIP-SPV/chipStar#1695; no upstream
// reports): in a global initializer, the in-tree SPIR-V backend aborts on a
// getelementptr over an addrspacecast of a global, and IGC stores 0 for it.
// Rewrites it as an addrspacecast of the getelementptr. Remove when both
// handle the original and #1703 is fixed.
class HipOffsetBeforeCastPass
    : public HipRequiredPassInfoMixin<HipOffsetBeforeCastPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    // Literal indices: rewriting one entry then cannot change, and free,
    // another.
    auto IsOffset = [](User *U, User *CE) {
      auto *GEP = dyn_cast<GEPOperator>(U);
      return GEP && isa<ConstantExpr>(U) && GEP->getPointerOperand() == CE &&
             all_of(GEP->indices(),
                    [](Value *Idx) { return isa<ConstantInt>(Idx); });
    };
    auto IsCast = [](User *U) {
      auto *CE = dyn_cast<ConstantExpr>(U);
      return CE && CE->getOpcode() == Instruction::AddrSpaceCast;
    };
    SmallVector<std::pair<GlobalVariable *, GEPOperator *>, 8> Work;
    for (GlobalVariable &GV : M.globals()) {
      GV.removeDeadConstantUsers();
      // Retype GV only if a rewrite below reaches a constant.
      if (none_of(GV.users(), [&](User *CE) {
            return IsCast(CE) && any_of(CE->users(), [&](User *U) {
                     return IsOffset(U, CE) &&
                            !all_of(U->users(), IsaPred<Instruction>);
                   });
          }) ||
          !retypeToBytes(GV))
        continue;
      for (User *CE : GV.users())
        if (IsCast(CE))
          for (User *U : CE->users())
            if (IsOffset(U, CE))
              Work.emplace_back(&GV, cast<GEPOperator>(U));
    }
    for (auto [GV, GEP] : Work) {
      SmallVector<Value *, 4> Idx(GEP->idx_begin(), GEP->idx_end());
      Constant *Offset = ConstantExpr::getGetElementPtr(
          GEP->getSourceElementType(), GV, Idx, GEP->getNoWrapFlags(),
          GEP->getInRange());
      // Direct instruction operands compile fine with the original shape.
      GEP->replaceUsesWithIf(
          ConstantExpr::getAddrSpaceCast(Offset, GEP->getType()),
          [](Use &U) { return !isa<Instruction>(U.getUser()); });
    }
    return Work.empty() ? PreservedAnalyses::all() : PreservedAnalyses::none();
  }
  static bool isRequired() { return true; }
};

// Insert a helper that adds a pass with HipVerify validation
template <typename PassT>
static void
addPassWithVerification(ModulePassManager &MPM, PassT &&P,
                        const std::string &Name,
                        bool Verify = HipVerifyPass::isVerificationEnabled()) {
  MPM.addPass(std::forward<PassT>(P));
  if (Verify) {
    // Use HipVerify pass with the name of the pass that just ran (no summary printing)
    // This will always run even if the previous pass failed
    MPM.addPass(HipVerifyPass(Name, false));
  }
}

static void addFullLinkTimePasses(ModulePassManager &MPM) {
  MPM.addPass(HipFixOpenCLMDPass()); // must be first or else we get OCL Version mismatch

#ifndef CHIP_KEEP_KERNEL_DEBUG_INFO
  // No SPIR-V producer emits debug information our consumers accept, so drop it
  // up front unless the build opted in (-DCHIP_KEEP_KERNEL_DEBUG_INFO=ON, which
  // only makes sense on Intel Data Center GPU Max). Doing it here also spares
  // the passes below from keeping debug metadata consistent as they erase
  // globals and functions. See HipStripDebugInfo.cpp.
  MPM.addPass(HipStripDebugInfoPass());
#endif

  // Clear any previous results at the start of a new pipeline
  HipVerifyPass::clearResults();

  // Initial verification
  MPM.addPass(HipVerifyPass("Pre-HIP passes", false)); // false = don't print summary yet

  // Use HipVerify for intermediate passes without printing summary
  addPassWithVerification(MPM, HipSanityChecksPass(), "HipSanityChecksPass");

  /// For extracting name expression to lowered name expressions (hiprtc).
  addPassWithVerification(MPM, HipEmitLoweredNamesPass(), "HipEmitLoweredNamesPass");

  // Remove attributes that may prevent the device code from being optimized.
  addPassWithVerification(MPM, RemoveNoInlineOptNoneAttrsPass(), "RemoveNoInlineOptNoneAttrsPass");

  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipLowerSwitchPass()), "HipLowerSwitchPass");

  // Before HipDynMem, which cannot rewrite a shared address in an initializer.
  addPassWithVerification(MPM, HipSharedAddrLocalInitPass(),
                          "HipSharedAddrLocalInitPass");

  // Run a collection of passes run at device link time.
  addPassWithVerification(MPM, HipDynMemExternReplaceNewPass(), "HipDynMemExternReplaceNewPass");
  // Should be after the HipDynMemExternReplaceNewPass which relies on detecting
  // dynamic shared memories being modeled as zero length arrays.
  addPassWithVerification(MPM, HipLowerZeroLengthArraysPass(), "HipLowerZeroLengthArraysPass");

  // Prepare device code for texture function lowering which does not yet work
  // on non-inlined code and local variables of hipTextureObject_t type.
  addPassWithVerification(MPM, RemoveNoInlineOptNoneAttrsPass(), "RemoveNoInlineOptNoneAttrsPass-2");
  // Increase getInlineParams argument for more aggressive inlining.
  addPassWithVerification(MPM, ModuleInlinerWrapperPass(getInlineParams(1000)), "ModuleInlinerWrapperPass");
#if LLVM_VERSION_MAJOR < 14
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(SROA()), "SROA");
#elif LLVM_VERSION_MAJOR < 16
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(SROAPass()), "SROAPass");
#else
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(SROAPass(SROAOptions::PreserveCFG)), "SROAPass-PreserveCFG");
#endif

  addPassWithVerification(MPM, HipTextureLoweringPass(), "HipTextureLoweringPass");

  // TODO: Update printf pass for HIP-Clang 14+. It now triggers an assert:
  //
  //  Assertion `isa<X>(Val) && "cast<Ty>() argument of incompatible type!"'
  //  failed.
  addPassWithVerification(MPM, HipPrintfToOpenCLPrintfPass(), "HipPrintfToOpenCLPrintfPass");
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipLowerMemsetPass()), "HipLowerMemsetPass");
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipLowerFPAtomicMinMaxPass()), "HipLowerFPAtomicMinMaxPass");
  // OpenCL SPIR-V consumers implement 32 and 64 bit atomics only; rewrite 8
  // and 16 bit ones onto their containing word. Runs after the fmin / fmax
  // expansion so the i16 cmpxchg it produces for half gets lowered too, and
  // before InferAddressSpaces so the word address it forms with a GEP is
  // still narrowed to the global or local address space.
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipLowerSubwordAtomicsPass()), "HipLowerSubwordAtomicsPass");
  addPassWithVerification(MPM, HipLowerRoundIntrinsicsPass(), "HipLowerRoundIntrinsicsPass");
  addPassWithVerification(MPM, HipAbortPass(), "HipAbortPass");
  // This pass must appear after HipDynMemExternReplaceNewPass.
  addPassWithVerification(MPM, HipGlobalVariablesPass(), "HipGlobalVariablesPass");
  addPassWithVerification(MPM, HipOffsetBeforeCastPass(), "HipOffsetBeforeCastPass");

  // This pass must be last one that modifies kernel parameter list.
  addPassWithVerification(MPM, HipKernelArgSpillerPass(), "HipKernelArgSpillerPass");

  // After the spiller, so the pin lands on the kernels it creates.
  addPassWithVerification(MPM, HipWarpsPass(), "HipWarpsPass");

  // After every pass that can create a copy, so a zero length one it emits is
  // erased too, and before the DCE below, so the address computations feeding
  // an erased llvm.prefetch go with it.
  addPassWithVerification(MPM, HipLowerHintIntrinsicsPass(),
                          "HipLowerHintIntrinsicsPass");

  // Remove dead code left over by HIP lowering passes and kept alive by
  // llvm.used and llvm.compiler.used intrinsic variable.
  addPassWithVerification(MPM, HipStripUsedIntrinsicsPass(), "HipStripUsedIntrinsicsPass");

  // Internalize all __device__ functions (spir_kernels) so the follow-up DCE
  // passes cleans-ups the unused ones.
  addPassWithVerification(MPM, InternalizePass(preserveDuringInternalize), "InternalizePass");
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(DCEPass()), "DCEPass");
  addPassWithVerification(MPM, GlobalDCEPass(), "GlobalDCEPass");

  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(InferAddressSpacesPass(4u)), "InferAddressSpacesPass");

  // Move vtable function pointers into the generic address space. Runs after
  // inlining and InferAddressSpaces so it only sees the indirect calls that
  // genuinely survive into the SPIR-V module.
  addPassWithVerification(MPM, HipFunctionPointerASPass(), "HipFunctionPointerASPass");

  addPassWithVerification(MPM, HipIGBADetectorPass(), "HipIGBADetectorPass");

  // A volatile global access carries CUDA's ld.volatile / st.volatile meaning
  // (an access that bypasses the core's cache) and SPIR-V's Volatile memory
  // operand does not, so rewrite them into relaxed device-scope atomics, which
  // the SPIR-V producers emit as OpAtomicLoad / OpAtomicStore.
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipLowerVolatileAccessesPass()), "HipLowerVolatileAccessesPass");

  // Fix InvalidBitWidth errors due to non-standard integer types
  addPassWithVerification(MPM, HipPromoteIntsPass(), "HipPromoteIntsPass");

  // Expand llvm.{u,s}mul.with.overflow, which clang emits for device-side
  // array-new and __builtin_mul_overflow. Left in place, the backend lane
  // emits OpUMulExtended, which IGC rejects with an undefined reference to
  // _Z20__spirv_UMulExtendedll, and the translator lane cannot lower the
  // signed intrinsic at all. Either failure takes down every kernel in the
  // module.
  addPassWithVerification(MPM, HipLowerOverflowIntrinsicsPass(),
                          "HipLowerOverflowIntrinsicsPass");

  // WORKAROUND(CHIP-SPV/chipStar#1577, llvm/llvm-project#217948): LLVM 23's
  // SROA folds a struct of pointers into <N x ptr>, which the in-tree backend
  // asserts on and llvm-spirv will not translate without
  // SPV_INTEL_masked_gather_scatter, an extension current IGC rejects. Carry
  // such values as integer vectors instead. Remove once the SPIR-V path
  // handles <N x ptr> itself.
  addPassWithVerification(MPM, HipLowerPointerVectorsPass(),
                          "HipLowerPointerVectorsPass");

  // Must be last: removes __chip_*/__hip_* globals and stubs their users.
  // Runs after HipIGBADetectorPass which creates __chip_module_has_no_IGBAs.
  addPassWithVerification(MPM, HipCleanupPass(), "HipCleanupPass");

#ifdef CHIP_LLVM_USE_INTERGRATED_SPIRV
  // After every pass that creates or removes FP operations or calls.
  addPassWithVerification(MPM, HipFPContractPass(), "HipFPContractPass");
  // WORKAROUND(CHIP-SPV/chipStar#1958, llvm/llvm-project (not yet reported)): see the pass.
  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(HipLowerMemmovePass()),
      "HipLowerMemmovePass");
#endif

  // Steers SPIR-V emission away from an access chain form IGC miscompiles.
  // Runs last so nothing downstream reintroduces the canonicalized shape.
  addPassWithVerification(MPM, HipCanonicalizeGEPPass(),
                          "HipCanonicalizeGEPPass");

  // WORKAROUND(CHIP-SPV/chipStar#1891, KhronosGroup/SPIRV-LLVM-Translator#3918, llvm/llvm-project#209232): see the pass.
  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(HipLowerSitofpI1Pass()),
      "HipLowerSitofpI1Pass");

  // WORKAROUND(CHIP-SPV/chipStar#1896, intel/intel-graphics-compiler (not yet reported)): see the pass.
  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(HipFloatShufflePass()),
      "HipFloatShufflePass");

#ifndef CHIP_MALI_GPU_WORKAROUNDS
  // WORKAROUND(CHIP-SPV/chipStar#1978, intel/intel-graphics-compiler: no upstream issue yet): see the pass.
  // After HipFloatShufflePass, so float shuffles shift as floats. Not on Mali,
  // whose clLinkProgram cannot link the rtdevlib.
  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(HipShuffleShiftPass()),
      "HipShuffleShiftPass");
#endif

  // WORKAROUND(CHIP-SPV/chipStar#1680, KhronosGroup/SPIRV-LLVM-Translator#3866): llvm-spirv emits one OpPhi entry per LLVM phi entry, duplicating predecessors. Remove when the pinned llvm_release branch includes #3866.
  // Last CFG change before SPIR-V emission, so nothing merges the forwarding blocks back.
  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(HipCoalesceDuplicatePhiPredsPass()),
      "HipCoalesceDuplicatePhiPredsPass");

  // Final verification pass with summary printing
  MPM.addPass(HipVerifyPass("Post-HIP passes", true)); // true = print final summary
}

#if LLVM_VERSION_MAJOR < 14
#define PASS_ID "hip-link-time-passes"
#else
#define PASS_ID "hip-post-link-passes"
#endif

extern "C" ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-passes", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_ID) {
                    addFullLinkTimePasses(MPM);
                    return true;
                  }
                  // Register IR-only validation pass as standalone (legacy - use hip-verify instead)
                  if (Name == "ir-validate") {
                    MPM.addPass(HipVerifyPass("IR validation"));
                    return true;
                  }
                  // Register separate SPIR-V validation pass as standalone (legacy - use hip-verify instead)
                  if (Name == "spirv-validate") {
                    MPM.addPass(HipVerifyPass("SPIR-V validation"));
                    return true;
                  }
                  // Register merged IR+SPIR-V validation pass as standalone (legacy - use hip-verify instead)
                  if (Name == "ir-spirv-validate") {
                    MPM.addPass(HipVerifyPass("IR+SPIR-V validation"));
                    return true;
                  }
                  // Register the overflow intrinsic lowering as standalone,
                  // which makes it directly testable with opt.
                  if (Name == "hip-lower-overflow-intrinsics") {
                    MPM.addPass(HipLowerOverflowIntrinsicsPass());
                    return true;
                  }
                  // Same for the pointer-vector lowering, so the workaround
                  // in #1577 can be tested with opt directly.
                  if (Name == "hip-lower-pointer-vectors") {
                    MPM.addPass(HipLowerPointerVectorsPass());
                    return true;
                  }
                  // Register the 8 and 16 bit atomic lowering as standalone,
                  // which makes it directly testable with opt.
                  if (Name == "hip-lower-subword-atomics") {
                    MPM.addPass(createModuleToFunctionPassAdaptor(
                        HipLowerSubwordAtomicsPass()));
                    return true;
                  }
                  // Register the volatile access lowering as standalone,
                  // which makes it directly testable with opt.
                  if (Name == "hip-lower-volatile-accesses") {
                    MPM.addPass(createModuleToFunctionPassAdaptor(
                        HipLowerVolatileAccessesPass()));
                    return true;
                  }
                  // Register the vtable function pointer address space pass
                  // as standalone, which makes it directly testable with opt.
                  if (Name == "hip-function-pointer-as") {
                    MPM.addPass(HipFunctionPointerASPass());
                    return true;
                  }
                  // Register the hint intrinsic lowering as standalone,
                  // which makes it directly testable with opt.
                  if (Name == "hip-lower-hint-intrinsics") {
                    MPM.addPass(HipLowerHintIntrinsicsPass());
                    return true;
                  }
                  // Register SPIR-V function reorder pass as standalone
                  if (Name == "hip-spirv-function-reorder") {
                    MPM.addPass(HipSpirvFunctionReorderPass());
                    return true;
                  }
                  // Register unified HipVerify pass
                  if (Name == "hip-verify") {
                    MPM.addPass(HipVerifyPass());
                    return true;
                  }
                  return false;
                });
          }};
}
