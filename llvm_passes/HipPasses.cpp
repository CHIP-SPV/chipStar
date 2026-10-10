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
#include "HipDefrost.h"
#include "HipDynMem.h"
#include "HipStripDebugInfo.h"
#include "HipStripUsedIntrinsics.h"
#include "HipWarps.h"
#include "HipPrintf.h"
#include "HipGlobalVariables.h"
#include "HipTextureLowering.h"
#include "../src/common.hh"
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
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "PassPluginCompat.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LowerAtomic.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Utils/Local.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/StructurizeCFG.h"
#include "llvm/Transforms/Utils/LowerSwitch.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/TargetParser/Triple.h"

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
    : public HipPassInfoMixin<RemoveNoInlineOptNoneAttrsPass> {
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
class HipFixOpenCLMDPass : public HipPassInfoMixin<HipFixOpenCLMDPass> {
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
    : public HipPassInfoMixin<HipSharedAddrLocalInitPass> {
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
class HipFPContractPass : public HipPassInfoMixin<HipFPContractPass> {
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
#endif

// WORKAROUND(CHIP-SPV/chipStar#1891, KhronosGroup/SPIRV-LLVM-Translator#3918): translator lowers sitofp i1 true to +1.0. Remove when the fix is in the translator release branch chipStar builds.
// WORKAROUND(CHIP-SPV/chipStar#1891, llvm/llvm-project#209232): the in-tree
// backend does the same. Remove when the fix is in the LLVM release chipStar
// builds. Delete the pass only once CanarySitofpI1Translator and
// CanarySitofpI1Backend both fire for every supported LLVM version.
// Rewrites sitofp i1 %c into select %c, -1.0, 0.0.
class HipLowerSitofpI1Pass : public PassInfoMixin<HipLowerSitofpI1Pass> {
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
                                     ConstantFP::get(Ty, 0.0), "", Cast);
      Sel->takeName(Cast);
      Cast->replaceAllUsesWith(Sel);
      Cast->eraseFromParent();
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
    : public HipPassInfoMixin<HipOffsetBeforeCastPass> {
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
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipDefrostPass()), "HipDefrostPass");
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

  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(InferAddressSpacesPass(4u)),
      "InferAddressSpacesPass");

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
#endif

  // Steers SPIR-V emission away from an access chain form IGC miscompiles.
  // Runs last so nothing downstream reintroduces the canonicalized shape.
  addPassWithVerification(MPM, HipCanonicalizeGEPPass(),
                          "HipCanonicalizeGEPPass");

  // WORKAROUND(CHIP-SPV/chipStar#1891, KhronosGroup/SPIRV-LLVM-Translator#3918, llvm/llvm-project#209232): see the pass.
  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(HipLowerSitofpI1Pass()),
      "HipLowerSitofpI1Pass");

  // WORKAROUND(CHIP-SPV/chipStar#1680, KhronosGroup/SPIRV-LLVM-Translator#3866): llvm-spirv emits one OpPhi entry per LLVM phi entry, duplicating predecessors. Remove when the pinned llvm_release branch includes #3866.
  // Last CFG change before SPIR-V emission, so nothing merges the forwarding blocks back.
  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(HipCoalesceDuplicatePhiPredsPass()),
      "HipCoalesceDuplicatePhiPredsPass");

  // Final verification pass with summary printing
  MPM.addPass(HipVerifyPass("Post-HIP passes", true)); // true = print final summary
}

// The native Vulkan pipeline; it needs LLVM 24's SPIR-V shader intrinsics.
#if LLVM_VERSION_MAJOR >= 24
// Erases globals nothing uses. chipStar's runtime globals are
// externally_initialized, which keeps Internalize and GlobalDCE off them, and
// Vulkan has no linking to resolve them.
class HipDropUnusedGlobalsPass
    : public HipPassInfoMixin<HipDropUnusedGlobalsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool Changed = false;
    for (GlobalVariable &GV : make_early_inc_range(M.globals())) {
      GV.removeDeadConstantUsers();
      if (GV.use_empty() && !GV.getName().starts_with("llvm.")) {
        GV.eraseFromParent();
        Changed = true;
      }
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Vulkan allows OpFDiv an error of 2.5 ulp and Mesa's ANV divides through a
// reciprocal; HIP division is correctly rounded unless fast math allows
// otherwise. Devices may flush subnormals, so the operands are split by
// integer operations into significands in [1, 2) and exponents, their quotient
// is refined with FMA residual steps, and the result is assembled, subnormal
// rounding included, from integers.
class HipVulkanPreciseDivPass
    : public HipPassInfoMixin<HipVulkanPreciseDivPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    auto Arcp = [](Value *V) {
      if (auto *I = dyn_cast<Instruction>(V))
        I->setHasAllowReciprocal(true);
      return V;
    };
    SmallVector<BinaryOperator *, 8> Divs;
    for (Instruction &I : instructions(F))
      if (auto *BO = dyn_cast<BinaryOperator>(&I);
          BO && BO->getOpcode() == Instruction::FDiv &&
          !BO->hasAllowReciprocal() &&
          (BO->getType()->getScalarType()->isFloatTy() ||
           BO->getType()->getScalarType()->isDoubleTy()))
        Divs.push_back(BO);
    for (BinaryOperator *D : Divs) {
      IRBuilder<> B(D);
      Type *T = D->getType();
      bool IsDouble = T->getScalarType()->isDoubleTy();
      unsigned Bits = IsDouble ? 64 : 32, Mant = IsDouble ? 52 : 23;
      uint64_t Bias = IsDouble ? 1023 : 127, MantMask = (1ull << Mant) - 1;
      Type *IT = T->getWithNewType(B.getIntNTy(Bits));
      auto C = [&](double V) { return ConstantFP::get(T, V); };
      auto I = [&](uint64_t V) { return ConstantInt::get(IT, V); };
      auto Fma = [&](Value *X, Value *Y, Value *Z) {
        return B.CreateIntrinsic(Intrinsic::fma, {T}, {X, Y, Z});
      };
      Value *A = D->getOperand(0), *Bv = D->getOperand(1);
      Value *AI = B.CreateBitCast(A, IT), *BI = B.CreateBitCast(Bv, IT);
      Value *Sign =
          B.CreateAnd(B.CreateXor(AI, BI), I(1ull << (Bits - 1)));
      Value *AbsA = B.CreateAnd(AI, I((1ull << (Bits - 1)) - 1)),
            *AbsB = B.CreateAnd(BI, I((1ull << (Bits - 1)) - 1));
      // |x| = M * 2^(E - Bias) with M in [1, 2); a subnormal is converted from
      // its integer significand, which is exact and normal.
      auto Split = [&](Value *Abs, Value *&M, Value *&E) {
        Value *Sub = B.CreateICmpULT(Abs, I(1ull << Mant));
        Value *X = B.CreateSelect(
            Sub, B.CreateBitCast(B.CreateUIToFP(Abs, T), IT), Abs);
        Value *Field = B.CreateLShr(X, I(Mant));
        E = B.CreateSelect(Sub, B.CreateSub(Field, I(Bias + Mant - 1)), Field);
        M = B.CreateBitCast(B.CreateOr(B.CreateAnd(X, I(MantMask)),
                                       I(Bias << Mant)),
                            T);
      };
      Value *MA, *EA, *MB, *EB;
      Split(AbsA, MA, EA);
      Split(AbsB, MB, EB);
      Value *R0 = Arcp(B.CreateFDiv(C(1.0), MB));
      // One Newton step on the reciprocal: the residual step needs it close to
      // decide quotients just past a rounding midpoint.
      Value *R = Fma(Fma(B.CreateFNeg(MB), R0, C(1.0)), R0, R0);
      Value *Q = B.CreateFMul(MA, R);
      for (int Step = 0; Step < (IsDouble ? 2 : 1); ++Step)
        Q = Fma(Fma(B.CreateFNeg(Q), MB, MA), R, Q);
      // The residual of q is exact: if it exceeds half the gap to the
      // neighbour beyond it, or ties on an odd q, q is one ulp off.
      Value *QBits = B.CreateBitCast(Q, IT);
      Value *Res = Fma(B.CreateFNeg(Q), MB, MA);
      Value *Next = B.CreateBitCast(
          B.CreateSelect(B.CreateFCmpOGE(Res, C(0.0)),
                         B.CreateAdd(QBits, I(1)), B.CreateSub(QBits, I(1))),
          T);
      Value *AbsRes = B.CreateUnaryIntrinsic(Intrinsic::fabs, Res);
      Value *Half = B.CreateFMul(
          B.CreateFMul(
              B.CreateUnaryIntrinsic(Intrinsic::fabs, B.CreateFSub(Next, Q)),
              MB),
          C(0.5));
      Value *Off = B.CreateOr(
          B.CreateFCmpOGT(AbsRes, Half),
          B.CreateAnd(B.CreateFCmpOEQ(AbsRes, Half),
                      B.CreateTrunc(QBits, IT->getWithNewType(B.getInt1Ty()))));
      Q = B.CreateSelect(Off, Next, Q);
      QBits = B.CreateBitCast(Q, IT);
      Res = Fma(B.CreateFNeg(Q), MB, MA);
      // The result's biased exponent; q is in (0.5, 2).
      Value *Diff = B.CreateSub(EA, EB);
      Value *EF = B.CreateAdd(B.CreateLShr(QBits, I(Mant)), Diff);
      Value *Normal = B.CreateAdd(QBits, B.CreateShl(Diff, I(Mant)));
      // Subnormal: shift the significand right, rounding to nearest even with
      // the residual as the bits below q.
      Value *Sh = B.CreateBinaryIntrinsic(
          Intrinsic::smin,
          B.CreateBinaryIntrinsic(Intrinsic::smax, B.CreateSub(I(1), EF), I(1)),
          I(Mant + 2));
      Value *Sig = B.CreateOr(B.CreateAnd(QBits, I(MantMask)), I(1ull << Mant));
      Value *Kept = B.CreateLShr(Sig, Sh);
      Value *Rem = B.CreateAnd(Sig, B.CreateSub(B.CreateShl(I(1), Sh), I(1)));
      Value *HalfI = B.CreateShl(I(1), B.CreateSub(Sh, I(1)));
      Value *Up = B.CreateOr(
          B.CreateICmpUGT(Rem, HalfI),
          B.CreateAnd(
              B.CreateICmpEQ(Rem, HalfI),
              B.CreateOr(B.CreateFCmpOGT(Res, C(0.0)),
                         B.CreateAnd(B.CreateFCmpOEQ(Res, C(0.0)),
                                     B.CreateTrunc(Kept, IT->getWithNewType(
                                                             B.getInt1Ty()))))));
      Value *Subnormal = B.CreateAdd(Kept, B.CreateZExt(Up, IT));
      Value *Inf = I((2 * Bias + 1) << Mant);
      Value *Mag = B.CreateSelect(
          B.CreateICmpSGE(EF, I(2 * Bias + 1)), Inf,
          B.CreateSelect(B.CreateICmpSGT(EF, I(0)), Normal, Subnormal));
      // Zeros and infinities by class; NaN from the plain division.
      Value *AZero = B.CreateICmpEQ(AbsA, I(0)),
            *BZero = B.CreateICmpEQ(AbsB, I(0)),
            *AInf = B.CreateICmpEQ(AbsA, Inf), *BInf = B.CreateICmpEQ(AbsB, Inf);
      Mag = B.CreateSelect(B.CreateOr(AInf, BZero), Inf,
                           B.CreateSelect(B.CreateOr(AZero, BInf), I(0), Mag));
      Value *NaN = B.CreateOr(
          B.CreateOr(B.CreateICmpUGT(AbsA, Inf), B.CreateICmpUGT(AbsB, Inf)),
          B.CreateOr(B.CreateAnd(AZero, BZero), B.CreateAnd(AInf, BInf)));
      Q = B.CreateSelect(NaN, Arcp(B.CreateFDiv(A, Bv)),
                         B.CreateBitCast(B.CreateOr(Sign, Mag), T));
      D->replaceAllUsesWith(Q);
      D->eraseFromParent();
    }
    // GLSL Sqrt is 1/inversesqrt; the same split gives a correctly rounded one.
    SmallVector<IntrinsicInst *, 8> Sqrts;
    for (Instruction &I : instructions(F))
      if (auto *II = dyn_cast<IntrinsicInst>(&I);
          II && II->getIntrinsicID() == Intrinsic::sqrt &&
          !II->hasApproxFunc() &&
          (II->getType()->getScalarType()->isFloatTy() ||
           II->getType()->getScalarType()->isDoubleTy()))
        Sqrts.push_back(II);
    for (IntrinsicInst *Sq : Sqrts) {
      IRBuilder<> B(Sq);
      Type *T = Sq->getType();
      bool IsDouble = T->getScalarType()->isDoubleTy();
      unsigned Bits = IsDouble ? 64 : 32, Mant = IsDouble ? 52 : 23;
      uint64_t Bias = IsDouble ? 1023 : 127, MantMask = (1ull << Mant) - 1;
      Type *IT = T->getWithNewType(B.getIntNTy(Bits));
      auto C = [&](double V) { return ConstantFP::get(T, V); };
      auto I = [&](uint64_t V) { return ConstantInt::get(IT, V); };
      auto Fma = [&](Value *X, Value *Y, Value *Z) {
        return B.CreateIntrinsic(Intrinsic::fma, {T}, {X, Y, Z});
      };
      Value *X = Sq->getArgOperand(0);
      Value *XI = B.CreateBitCast(X, IT);
      Value *Abs = B.CreateAnd(XI, I((1ull << (Bits - 1)) - 1));
      // x = M * 2^E with M in [1, 4) and E even.
      Value *Sub = B.CreateICmpULT(Abs, I(1ull << Mant));
      Value *N =
          B.CreateSelect(Sub, B.CreateBitCast(B.CreateUIToFP(Abs, T), IT), Abs);
      Value *Field = B.CreateLShr(N, I(Mant));
      Value *E = B.CreateSub(
          B.CreateSelect(Sub, B.CreateSub(Field, I(Bias + Mant - 1)), Field),
          I(Bias));
      Value *Odd = B.CreateTrunc(E, IT->getWithNewType(B.getInt1Ty()));
      Value *M = B.CreateBitCast(
          B.CreateOr(B.CreateAnd(N, I(MantMask)), I(Bias << Mant)), T);
      M = B.CreateSelect(Odd, B.CreateFMul(M, C(2.0)), M);
      Value *S = B.CreateUnaryIntrinsic(Intrinsic::sqrt, M);
      if (auto *SI = dyn_cast<Instruction>(S))
        SI->setHasApproxFunc(true);
      // Newton steps, then one ulp either way by the exact residual.
      for (int Step = 0; Step < (IsDouble ? 2 : 1); ++Step) {
        Value *H = Arcp(B.CreateFDiv(C(0.5), S));
        S = Fma(Fma(B.CreateFNeg(S), S, M), H, S);
      }
      Value *SBits = B.CreateBitCast(S, IT);
      Value *Down = B.CreateBitCast(B.CreateSub(SBits, I(1)), T);
      Value *Up = B.CreateBitCast(B.CreateAdd(SBits, I(1)), T);
      S = B.CreateSelect(
          B.CreateFCmpOLE(Fma(B.CreateFNeg(Down), S, M), C(0.0)), Down, S);
      S = B.CreateSelect(B.CreateFCmpOGT(Fma(B.CreateFNeg(Up), S, M), C(0.0)),
                         Up, S);
      Value *R = B.CreateBitCast(
          B.CreateAdd(B.CreateBitCast(S, IT),
                      B.CreateShl(B.CreateAShr(E, I(1)), I(Mant))),
          T);
      // Zeros, +inf and NaN are their own root; other negatives have none.
      Value *Inf = I((2 * Bias + 1) << Mant);
      Value *Neg = B.CreateAnd(B.CreateICmpSLT(XI, I(0)),
                               B.CreateAnd(B.CreateICmpNE(Abs, I(0)),
                                           B.CreateICmpULE(Abs, Inf)));
      R = B.CreateSelect(
          Neg, ConstantFP::getNaN(T),
          B.CreateSelect(
              B.CreateOr(B.CreateICmpEQ(Abs, I(0)), B.CreateICmpUGE(Abs, Inf)),
              X, R));
      Sq->replaceAllUsesWith(R);
      Sq->eraseFromParent();
    }
    return Divs.empty() && Sqrts.empty() ? PreservedAnalyses::all()
                                         : PreservedAnalyses::none();
  }
};

// Vulkan memory holds no pointers, so a pointer-valued __device__ variable
// that device code only reads (`__device__ float *P = Buf;`) is replaced by
// its initializer and the variable is erased.
class HipVulkanFoldPointerGlobalsPass
    : public HipPassInfoMixin<HipVulkanFoldPointerGlobalsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool Changed = false;
    for (GlobalVariable &GV : make_early_inc_range(M.globals())) {
      if (!GV.getValueType()->isPointerTy() || !GV.hasInitializer() ||
          GV.getInitializer()->isNullValue() ||
          GV.getAddressSpace() != SPIRV_CROSSWORKGROUP_AS)
        continue;
      GV.removeDeadConstantUsers();
      SmallVector<LoadInst *, 4> Loads;
      SmallVector<Instruction *, 4> Casts;
      std::function<bool(Value *)> OnlyLoads = [&](Value *V) {
        return all_of(V->users(), [&](User *U) {
          if (isa<AddrSpaceCastOperator>(U)) {
            if (auto *I = dyn_cast<Instruction>(U))
              Casts.push_back(I);
            return OnlyLoads(U);
          }
          auto *L = dyn_cast<LoadInst>(U);
          return L && !L->isVolatile() && L->getType() == GV.getValueType() &&
                 (Loads.push_back(L), true);
        });
      };
      if (!OnlyLoads(&GV) || Loads.empty())
        continue;
      for (LoadInst *L : Loads) {
        L->replaceAllUsesWith(GV.getInitializer());
        L->eraseFromParent();
      }
      for (Instruction *C : reverse(Casts))
        C->eraseFromParent();
      // No storage left: host symbol access fails instead of going stale.
      GV.removeDeadConstantUsers();
      GV.eraseFromParent();
      Changed = true;
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Erases the abort flag and message when no kernel can abort, and the printf
// buffer when no kernel prints, which tells the runtime not to read them back.
class HipVulkanDropUnusedAbortFlagPass
    : public HipPassInfoMixin<HipVulkanDropUnusedAbortFlagPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool Changed = false;
    for (const char *Name : {ChipDeviceAbortFlagName, ChipDeviceAbortMsgName,
                             "__hipspv_printf_buf"})
      if (GlobalVariable *GV = M.getGlobalVariable(Name)) {
        GV->removeDeadConstantUsers();
        if (GV->use_empty()) {
          GV->eraseFromParent();
          Changed = true;
        }
      }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Mesa's ANV on Intel Arc has neither shaderSharedFloat{32,64}AtomicAdd nor
// shaderSharedInt64Atomics, so a shared float atomic add cannot be a native
// atomic or a compare-and-swap loop: a work-group lock serializes it. Each lane
// retries the lock from a loop rather than spinning in the branch, which would
// deadlock the lanes of one subgroup.
class HipVulkanLockSharedFloatAtomicsPass
    : public HipPassInfoMixin<HipVulkanLockSharedFloatAtomicsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    SmallVector<AtomicRMWInst *, 8> Work;
    for (Function &F : M)
      for (Instruction &I : instructions(F))
        if (auto *RMW = dyn_cast<AtomicRMWInst>(&I);
            RMW && RMW->getPointerAddressSpace() == SPIRV_WORKGROUP_AS &&
            RMW->getType()->isFloatingPointTy())
          Work.push_back(RMW);
    if (Work.empty())
      return PreservedAnalyses::all();
    LLVMContext &C = M.getContext();
    Type *I32 = Type::getInt32Ty(C);
    auto *Lock = new GlobalVariable(
        M, I32, false, GlobalValue::InternalLinkage, PoisonValue::get(I32),
        "__chip_fatomic_lock", nullptr, GlobalValue::NotThreadLocal,
        SPIRV_WORKGROUP_AS);
    Lock->setAlignment(Align(4));
    SyncScope::ID WG = C.getOrInsertSyncScopeID("workgroup");
    SmallSetVector<Function *, 4> Kernels;
    for (AtomicRMWInst *RMW : Work) {
      Function *F = RMW->getFunction();
      Kernels.insert(F);
      BasicBlock *Pre = RMW->getParent();
      BasicBlock *Exit = Pre->splitBasicBlock(RMW, "fatomic.exit");
      Pre->getTerminator()->eraseFromParent();
      auto *Try = BasicBlock::Create(C, "fatomic.try", F, Exit);
      auto *Crit = BasicBlock::Create(C, "fatomic.crit", F, Exit);
      auto *Next = BasicBlock::Create(C, "fatomic.next", F, Exit);
      IRBuilder<> B(Pre);
      B.CreateBr(Try);
      B.SetInsertPoint(Try);
      Value *Got = B.CreateExtractValue(
          B.CreateAtomicCmpXchg(Lock, B.getInt32(0), B.getInt32(1), Align(4),
                                AtomicOrdering::Acquire,
                                AtomicOrdering::Monotonic, WG),
          1);
      B.CreateCondBr(Got, Crit, Next);
      B.SetInsertPoint(Crit);
      Value *Old = B.CreateLoad(RMW->getType(), RMW->getPointerOperand());
      Value *New = buildAtomicRMWValue(RMW->getOperation(), B, Old,
                                       RMW->getValOperand());
      B.CreateStore(New, RMW->getPointerOperand());
      B.CreateAtomicRMW(AtomicRMWInst::Xchg, Lock, B.getInt32(0), Align(4),
                        AtomicOrdering::Release, WG);
      B.CreateBr(Next);
      B.SetInsertPoint(Next);
      PHINode *Done = B.CreatePHI(B.getInt1Ty(), 2);
      Done->addIncoming(B.getTrue(), Crit);
      Done->addIncoming(B.getFalse(), Try);
      PHINode *Res = B.CreatePHI(RMW->getType(), 2);
      Res->addIncoming(Old, Crit);
      Res->addIncoming(PoisonValue::get(RMW->getType()), Try);
      B.CreateCondBr(Done, Exit, Try);
      RMW->replaceAllUsesWith(Res);
      RMW->eraseFromParent();
    }
    // The first work item clears the lock before anyone takes it.
    for (Function *F : Kernels) {
      IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());
      Value *Lid =
          B.CreateIntrinsic(I32, Intrinsic::spv_flattened_thread_id_in_group, {});
      Instruction *Then = SplitBlockAndInsertIfThen(
          B.CreateICmpEQ(Lid, B.getInt32(0)), B.GetInsertPoint(), false);
      IRBuilder<>(Then).CreateStore(B.getInt32(0), Lock);
      BasicBlock *After = Then->getSuccessor(0);
      IRBuilder<>(After, After->getFirstNonPHIIt())
          .CreateIntrinsic(Intrinsic::spv_group_memory_barrier_with_group_sync,
                           {});
    }
    return PreservedAnalyses::none();
  }
};

// Sums the i32 V over the work group through a shared counter, at the
// builder's position; every thread of the group must reach it.
static Value *blockSum(Module &M, IRBuilder<> &B, Value *V) {
  Type *I32 = B.getInt32Ty();
  auto *Counter = new GlobalVariable(
      M, I32, false, GlobalValue::InternalLinkage, PoisonValue::get(I32),
      "__chip_block_sum", nullptr, GlobalValue::NotThreadLocal, 3);
  auto Barrier = [&] {
    B.CreateIntrinsic(Intrinsic::spv_group_memory_barrier_with_group_sync, {});
  };
  Value *Lid =
      B.CreateIntrinsic(I32, Intrinsic::spv_flattened_thread_id_in_group, {});
  Barrier();
  Instruction *Then = SplitBlockAndInsertIfThen(
      B.CreateICmpEQ(Lid, B.getInt32(0)), &*B.GetInsertPoint(), false);
  IRBuilder<>(Then).CreateStore(B.getInt32(0), Counter);
  B.SetInsertPoint(&*B.GetInsertPoint());
  Barrier();
  B.CreateAtomicRMW(AtomicRMWInst::Add, Counter, V, MaybeAlign(4),
                    AtomicOrdering::Monotonic,
                    M.getContext().getOrInsertSyncScopeID("workgroup"));
  Barrier();
  Value *N = B.CreateLoad(I32, Counter);
  Barrier();
  return N;
}

// Replaces calls to chipStar's device library helpers that have no Vulkan
// device library implementation with the IR the SPIR-V backend lowers.
class HipVulkanLowerBuiltinsPass
    : public HipPassInfoMixin<HipVulkanLowerBuiltinsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    LLVMContext &C = M.getContext();
    bool Changed = false;
    auto Replace = [&](StringRef Name, auto Emit) {
      Function *F = M.getFunction(Name);
      if (!F)
        return;
      // The OpenCL device library body uses builtins Vulkan lacks.
      if (!F->isDeclaration())
        F->deleteBody();
      for (User *U : make_early_inc_range(F->users()))
        if (auto *CI = dyn_cast<CallInst>(U); CI && CI->getCalledFunction() == F) {
          IRBuilder<> B(CI);
          if (Value *V = Emit(B, CI))
            CI->replaceAllUsesWith(V);
          CI->eraseFromParent();
          Changed = true;
        }
    };
    // __syncthreads orders all memory, not just shared memory.
    Replace("__chip_syncthreads", [&](IRBuilder<> &B, CallInst *) -> Value * {
      B.CreateIntrinsic(Intrinsic::spv_all_memory_barrier_with_group_sync, {});
      return nullptr;
    });
    Replace("__chip_threadfence_block", [&](IRBuilder<> &B, CallInst *) -> Value * {
      B.CreateFence(AtomicOrdering::SequentiallyConsistent,
                    C.getOrInsertSyncScopeID("workgroup"));
      return nullptr;
    });
    for (StringRef Name : {"__chip_threadfence", "__chip_threadfence_system"})
      Replace(Name, [&](IRBuilder<> &B, CallInst *) -> Value * {
        B.CreateFence(AtomicOrdering::SequentiallyConsistent,
                      C.getOrInsertSyncScopeID("device"));
        return nullptr;
      });
    // Warp shuffles, __shfl[_up|_down|_xor](var, x, width), for every type.
    for (Function &F : make_early_inc_range(M.functions())) {
      StringRef N = F.getName();
      int Kind = N.starts_with("_Z6__shfl")       ? 0
                 : N.starts_with("_Z9__shfl_up")   ? 1
                 : N.starts_with("_Z11__shfl_down") ? 2
                 : N.starts_with("_Z10__shfl_xor") ? 3
                                                    : -1;
      if (Kind < 0 || F.arg_size() != 3)
        continue;
      Replace(N, [&, Kind](IRBuilder<> &B, CallInst *CI) -> Value * {
        Type *I32 = B.getInt32Ty();
        Value *Var = CI->getArgOperand(0);
        Value *X = B.CreateZExtOrTrunc(CI->getArgOperand(1), I32);
        Value *W = B.CreateZExtOrTrunc(CI->getArgOperand(2), I32);
        Value *Lane = B.CreateIntrinsic(
            I32, Intrinsic::spv_subgroup_local_invocation_id, {});
        Value *Mask = B.CreateSub(W, B.getInt32(1));
        Value *Seg = B.CreateAnd(Lane, B.CreateNot(Mask));
        Value *Src;
        if (Kind == 0) {
          Src = B.CreateAdd(Seg, B.CreateAnd(X, Mask));
        } else if (Kind == 1) {
          Value *Up = B.CreateSub(Lane, X);
          Src = B.CreateSelect(B.CreateICmpSLT(Up, Seg), Lane, Up);
        } else if (Kind == 2) {
          Value *Down = B.CreateAdd(Lane, X);
          Src = B.CreateSelect(
              B.CreateICmpUGE(B.CreateAdd(B.CreateAnd(Lane, Mask), X), W), Lane,
              Down);
        } else {
          Value *Xor = B.CreateXor(Lane, X);
          Src = B.CreateSelect(B.CreateICmpUGE(Xor, B.CreateAdd(Seg, W)), Lane,
                               Xor);
        }
        return B.CreateIntrinsic(Var->getType(), Intrinsic::spv_wave_readlane,
                                 {Var, Src});
      });
    }
    // OpenCL builtins clspv provides natively, and chipStar's helpers for them.
    auto Unary = [&](StringRef Name, Intrinsic::ID ID) {
      Replace(Name, [&, ID](IRBuilder<> &B, CallInst *CI) -> Value * {
        return B.CreateUnaryIntrinsic(ID, CI->getArgOperand(0));
      });
    };
    for (StringRef N : {"_Z4sqrtf", "_Z4sqrtd"})
      Unary(N, Intrinsic::sqrt);
    for (StringRef N : {"_Z4fabsf", "_Z4fabsd"})
      Unary(N, Intrinsic::fabs);
    for (StringRef N : {"_Z5rsqrtf", "_Z5rsqrtd"})
      Unary(N, Intrinsic::spv_rsqrt);
    for (StringRef N : {"_Z8copysignff", "_Z8copysigndd"})
      Replace(N, [&](IRBuilder<> &B, CallInst *CI) -> Value * {
        return B.CreateBinaryIntrinsic(Intrinsic::copysign, CI->getArgOperand(0),
                                       CI->getArgOperand(1));
      });
    for (StringRef N : {"_Z3absi", "_Z3absl"})
      Replace(N, [&](IRBuilder<> &B, CallInst *CI) -> Value * {
        return B.CreateBinaryIntrinsic(Intrinsic::abs, CI->getArgOperand(0),
                                       B.getFalse());
      });
    for (StringRef N : {"__chip_lrint_f32", "__chip_lrint_f64"})
      Replace(N, [&](IRBuilder<> &B, CallInst *CI) -> Value * {
        return B.CreateFPToSI(
            B.CreateUnaryIntrinsic(Intrinsic::rint, CI->getArgOperand(0)),
            CI->getType());
      });
    // chipStar's atomic helpers: __chip_atomic_<op>[_system]_<type>. Vulkan
    // has no system scope; device scope is the widest.
    for (Function &F : make_early_inc_range(M.functions())) {
      StringRef N = F.getName();
      if (!N.consume_front("__chip_atomic_"))
        continue;
      StringRef Op = N.take_until([](char Ch) { return Ch == '_'; });
      StringRef Ty = N.substr(N.rfind('_') + 1);
      bool Float = Ty.starts_with("f"), Signed = Ty == "i" || Ty == "l";
      std::optional<AtomicRMWInst::BinOp> RMW;
      if (Op == "add")
        RMW = Float ? AtomicRMWInst::FAdd : AtomicRMWInst::Add;
      else if (Op == "sub")
        RMW = AtomicRMWInst::Sub;
      else if (Op == "and")
        RMW = AtomicRMWInst::And;
      else if (Op == "or")
        RMW = AtomicRMWInst::Or;
      else if (Op == "xor")
        RMW = AtomicRMWInst::Xor;
      else if (Op == "exch")
        RMW = AtomicRMWInst::Xchg;
      else if (Op == "min")
        RMW = Float ? AtomicRMWInst::FMin
                    : Signed ? AtomicRMWInst::Min : AtomicRMWInst::UMin;
      else if (Op == "max")
        RMW = Float ? AtomicRMWInst::FMax
                    : Signed ? AtomicRMWInst::Max : AtomicRMWInst::UMax;
      else if (Op == "inc2")
        RMW = AtomicRMWInst::UIncWrap;
      else if (Op == "dec2")
        RMW = AtomicRMWInst::UDecWrap;
      else if (Op != "cmpxchg")
        continue;
      SyncScope::ID Device = C.getOrInsertSyncScopeID("device");
      Replace(F.getName(), [&, RMW, Device](IRBuilder<> &B, CallInst *CI) -> Value * {
        Value *Addr = CI->getArgOperand(0);
        if (auto *PTI = dyn_cast<PtrToIntInst>(Addr))
          Addr = PTI->getPointerOperand();
        else if (!Addr->getType()->isPointerTy())
          Addr = B.CreateIntToPtr(Addr, B.getPtrTy(4));
        if (RMW)
          return B.CreateAtomicRMW(*RMW, Addr, CI->getArgOperand(1), MaybeAlign(),
                                   AtomicOrdering::Monotonic, Device);
        Value *Pair = B.CreateAtomicCmpXchg(
            Addr, CI->getArgOperand(1), CI->getArgOperand(2), MaybeAlign(),
            AtomicOrdering::Monotonic, AtomicOrdering::Monotonic, Device);
        return B.CreateExtractValue(Pair, 0);
      });
    }
    // libclc leaves the builtins clspv provides natively undefined.
    Replace("_Z12__clc_mul_hijj", [&](IRBuilder<> &B, CallInst *CI) -> Value * {
      Value *P = B.CreateMul(B.CreateZExt(CI->getArgOperand(0), B.getInt64Ty()),
                             B.CreateZExt(CI->getArgOperand(1), B.getInt64Ty()));
      return B.CreateTrunc(B.CreateLShr(P, 32), B.getInt32Ty());
    });
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Lowers printf with a constant format to a record appended to
// __hipspv_printf_buf, which the runtime prints when it synchronizes. The
// buffer holds the words used, an unused word, the records and a last word that
// absorbs overflowing writes. A record is [words, format bytes, format words,
// per argument: tag, low word, high word], tags being 1 int, 5 64-bit int,
// 2 double, 3 pointer and 4 constant string, whose words follow.
class HipVulkanLowerPrintfPass
    : public HipPassInfoMixin<HipVulkanLowerPrintfPass> {
  static std::optional<StringRef> constString(Value *V) {
    auto *GV = dyn_cast<GlobalVariable>(V->stripPointerCasts());
    if (!GV || !GV->isConstant() || !GV->hasDefinitiveInitializer())
      return std::nullopt;
    if (GV->getInitializer()->isNullValue() &&
        GV->getValueType()->isArrayTy() &&
        GV->getValueType()->getArrayElementType()->isIntegerTy(8))
      return StringRef("");
    auto *CDA = dyn_cast<ConstantDataArray>(GV->getInitializer());
    if (!CDA || !CDA->isCString())
      return std::nullopt;
    return CDA->getAsCString();
  }

  // For a pointer that picks one of several string literals at run time (a
  // select, a phi, or a local array of them), an i32 saying which, mirroring
  // the pointer's data flow; Cands collects the strings.
  static Value *stringChoice(Value *V, SmallVectorImpl<StringRef> &Cands,
                             DenseMap<Value *, Value *> &Memo, int Depth = 0) {
    if (Depth > 8)
      return nullptr;
    if (auto It = Memo.find(V); It != Memo.end())
      return It->second;
    Type *I32 = Type::getInt32Ty(V->getContext());
    if (std::optional<StringRef> S = constString(V)) {
      Cands.push_back(*S);
      return Memo[V] = ConstantInt::get(I32, Cands.size() - 1);
    }
    if (auto *Sel = dyn_cast<SelectInst>(V)) {
      Value *T = stringChoice(Sel->getTrueValue(), Cands, Memo, Depth + 1);
      Value *F = stringChoice(Sel->getFalseValue(), Cands, Memo, Depth + 1);
      return T && F ? Memo[V] = SelectInst::Create(Sel->getCondition(), T, F,
                                                   "", Sel->getIterator())
                    : nullptr;
    }
    if (auto *Phi = dyn_cast<PHINode>(V)) {
      auto *NP = PHINode::Create(I32, Phi->getNumIncomingValues(), "",
                                 Phi->getIterator());
      Memo[V] = NP;
      for (auto [In, BB] : zip(Phi->incoming_values(), Phi->blocks())) {
        Value *C = stringChoice(In, Cands, Memo, Depth + 1);
        if (!C) {
          Memo.erase(V);
          NP->replaceAllUsesWith(PoisonValue::get(I32));
          NP->eraseFromParent();
          return nullptr;
        }
        NP->addIncoming(C, BB);
      }
      return NP;
    }
    // A load from a local array of string pointers: an i32 array mirrors it.
    auto *LI = dyn_cast<LoadInst>(V);
    auto *Slot = LI ? dyn_cast<AllocaInst>(getUnderlyingObject(
                          LI->getPointerOperand()))
                    : nullptr;
    if (!Slot)
      return nullptr;
    const DataLayout &DL = LI->getDataLayout();
    uint64_t N = DL.getTypeAllocSize(Slot->getAllocatedType()) / 8;
    if (N == 0 || N > 64)
      return nullptr;
    // Element index of a pointer into the slot.
    auto ElemIdx = [&](Value *P, Instruction *At) -> Value * {
      IRBuilder<> B(At);
      Value *Idx = B.getInt32(0);
      while (P != Slot) {
        if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(P)) {
          P = ASC->getPointerOperand();
          continue;
        }
        auto *G = dyn_cast<GEPOperator>(P);
        if (!G)
          return nullptr;
        Idx = B.CreateAdd(
            Idx, B.CreateTrunc(B.CreateLShr(emitGEPOffset(&B, DL, cast<User>(G)),
                                            3),
                               I32));
        P = G->getPointerOperand();
      }
      return Idx;
    };
    SmallVector<StoreInst *, 4> Stores;
    for (User *U : Slot->users()) {
      SmallVector<User *, 4> Work{U};
      while (!Work.empty()) {
        User *W = Work.pop_back_val();
        if (auto *SI = dyn_cast<StoreInst>(W))
          Stores.push_back(SI);
        else if (isa<GEPOperator, AddrSpaceCastOperator>(W))
          append_range(Work, W->users());
        else if (!isa<LoadInst>(W) &&
                 !(isa<IntrinsicInst>(W) &&
                   cast<IntrinsicInst>(W)->isLifetimeStartOrEnd()))
          return nullptr;
      }
    }
    auto *IdTy = ArrayType::get(I32, N);
    auto *IdSlot =
        new AllocaInst(IdTy, Slot->getAddressSpace(), "", Slot->getIterator());
    for (StoreInst *SI : Stores) {
      Value *Id = stringChoice(SI->getValueOperand(), Cands, Memo, Depth + 1);
      Value *EI = ElemIdx(SI->getPointerOperand(), SI);
      if (!Id || !EI)
        return nullptr;
      IRBuilder<> B(SI);
      B.CreateStore(Id, B.CreateInBoundsGEP(IdTy, IdSlot, {B.getInt32(0), EI}));
    }
    Value *EI = ElemIdx(LI->getPointerOperand(), LI);
    if (!EI)
      return nullptr;
    IRBuilder<> B(LI);
    return Memo[V] = B.CreateLoad(
               I32, B.CreateInBoundsGEP(IdTy, IdSlot, {B.getInt32(0), EI}));
  }

  static bool eraseIfWriteOnly(AllocaInst *AI) {
    if (!AI->getAllocatedType()->isArrayTy() ||
        !AI->getAllocatedType()->getArrayElementType()->isPointerTy())
      return false;
    SmallVector<Instruction *, 8> Users;
    SmallVector<Value *, 8> Work{AI};
    while (!Work.empty()) {
      Value *V = Work.pop_back_val();
      for (User *U : V->users()) {
        auto *UI = dyn_cast<Instruction>(U);
        if (!UI)
          return false;
        if (isa<GetElementPtrInst, AddrSpaceCastInst>(UI))
          Work.push_back(UI);
        else if (!(isa<StoreInst>(UI) &&
                   cast<StoreInst>(UI)->getPointerOperand() == V) &&
                 !(isa<LoadInst>(UI) && UI->use_empty()) &&
                 !(isa<IntrinsicInst>(UI) &&
                   cast<IntrinsicInst>(UI)->isLifetimeStartOrEnd()) &&
                 !isa<DbgInfoIntrinsic>(UI))
          return false;
        Users.push_back(UI);
      }
    }
    SmallPtrSet<Instruction *, 8> Dead;
    for (bool Progress = true; Progress;) {
      Progress = false;
      for (Instruction *U : Users)
        if (!Dead.contains(U) && U->use_empty()) {
          U->eraseFromParent();
          Dead.insert(U);
          Progress = true;
        }
    }
    if (AI->use_empty())
      AI->eraseFromParent();
    return true;
  }

  // Which arguments the format consumes with %s ('*' widths take one too).
  static SmallVector<bool, 8> stringArgs(StringRef Fmt) {
    SmallVector<bool, 8> Str;
    for (size_t P = 0; P < Fmt.size(); ++P) {
      if (Fmt[P] != '%')
        continue;
      if (++P < Fmt.size() && Fmt[P] == '%')
        continue;
      for (; P < Fmt.size() && !isAlpha(Fmt[P]); ++P)
        if (Fmt[P] == '*')
          Str.push_back(false);
      while (P < Fmt.size() && StringRef("hljztL").contains(Fmt[P]))
        ++P;
      if (P < Fmt.size())
        Str.push_back(Fmt[P] == 's');
    }
    return Str;
  }
  static void appendWords(IRBuilder<> &B, StringRef S,
                          SmallVectorImpl<Value *> &W) {
    for (size_t I = 0; I < S.size(); I += 4) {
      uint32_t Word = 0;
      for (size_t J = 0; J < 4 && I + J < S.size(); ++J)
        Word |= uint32_t(uint8_t(S[I + J])) << (8 * J);
      W.push_back(B.getInt32(Word));
    }
  }

public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    GlobalVariable *Buf = M.getNamedGlobal("__hipspv_printf_buf");
    if (!Buf)
      return PreservedAnalyses::all();
    // Unused when compiled, it may be unnamed_addr, which device variables
    // the runtime reads are not.
    Buf->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    Type *BufTy = Buf->getValueType();
    uint64_t Last = BufTy->getArrayNumElements() - 1;
    bool Changed = false;
    // __chip_vk_eprintf prints to stderr.
    SmallVector<CallInst *, 16> Calls;
    for (StringRef Name : {"printf", "__chip_vk_eprintf"})
      if (Function *F = M.getFunction(Name))
        for (User *U : F->users())
          if (auto *CI = dyn_cast<CallInst>(U); CI && CI->getCalledFunction() == F)
            Calls.push_back(CI);
    for (CallInst *CI : Calls) {
      StringRef Name = CI->getCalledFunction()->getName();
      std::optional<StringRef> Fmt = constString(CI->getArgOperand(0));
      if (!Fmt)
        continue;
      IRBuilder<> B(CI);
      SmallVector<Value *, 32> W{
          nullptr, B.getInt32(Fmt->size() | (Name == "printf" ? 0 : 1u << 31))};
      appendWords(B, *Fmt, W);
      SmallVector<bool, 8> StrArgs = stringArgs(*Fmt);
      for (auto [ArgNo, Arg] : enumerate(drop_begin(CI->args()))) {
        Value *A = Arg;
        Type *T = A->getType();
        if (std::optional<StringRef> S = constString(A)) {
          W.append({B.getInt32(4), B.getInt32(S->size()), B.getInt32(0)});
          appendWords(B, *S, W);
          continue;
        }
        SmallVector<StringRef, 4> Cands;
        DenseMap<Value *, Value *> Memo;
        if (T->isPointerTy())
          if (Value *Id = stringChoice(A, Cands, Memo)) {
            // Every candidate padded to the longest; the length picks one.
            SmallVector<SmallVector<Value *, 8>, 4> CW(Cands.size());
            size_t Words = 1;
            for (auto [K, Str] : enumerate(Cands)) {
              appendWords(B, Str, CW[K]);
              Words = std::max(Words, CW[K].size());
            }
            auto Pick = [&](function_ref<Value *(size_t)> Of) {
              Value *R = Of(0);
              for (size_t K = 1; K < Cands.size(); ++K)
                R = B.CreateSelect(B.CreateICmpEQ(Id, B.getInt32(K)), Of(K), R);
              return R;
            };
            W.append({B.getInt32(4),
                      Pick([&](size_t K) { return B.getInt32(Cands[K].size()); }),
                      B.getInt32(Words)});
            for (size_t I = 0; I < Words; ++I)
              W.push_back(Pick([&](size_t K) {
                return I < CW[K].size() ? CW[K][I] : B.getInt32(0);
              }));
            continue;
          }
        if (T->isPointerTy() && ArgNo < StrArgs.size() && StrArgs[ArgNo]) {
          // A string in device memory: up to 256 bytes, to its NUL.
          constexpr unsigned StrMax = 256, StrWords = StrMax / 4;
          Function *F = CI->getFunction();
          IRBuilder<> EB(&*F->getEntryBlock().getFirstInsertionPt());
          auto *WTy = ArrayType::get(B.getInt32Ty(), StrWords);
          AllocaInst *Words = EB.CreateAlloca(WTy);
          BasicBlock *Pre = CI->getParent();
          BasicBlock *After = Pre->splitBasicBlock(B.GetInsertPoint(), "str.copied");
          BasicBlock *Loop = BasicBlock::Create(M.getContext(), "str.copy", F, After);
          BasicBlock *Body = BasicBlock::Create(M.getContext(), "str.byte", F, After);
          Pre->getTerminator()->eraseFromParent();
          IRBuilder<> PB(Pre);
          PB.CreateStore(ConstantAggregateZero::get(WTy), Words);
          PB.CreateBr(Loop);
          IRBuilder<> LB(Loop);
          PHINode *I = LB.CreatePHI(LB.getInt32Ty(), 2);
          I->addIncoming(LB.getInt32(0), Pre);
          // At the cap, reload the last byte instead of reading past it.
          Value *Full = LB.CreateICmpEQ(I, LB.getInt32(StrMax));
          Value *Idx = LB.CreateSelect(Full, LB.getInt32(StrMax - 1), I);
          Value *Ch = LB.CreateLoad(LB.getInt8Ty(),
                                    LB.CreateInBoundsGEP(LB.getInt8Ty(), A, Idx));
          LB.CreateCondBr(LB.CreateOr(LB.CreateICmpEQ(Ch, LB.getInt8(0)), Full),
                          After, Body);
          IRBuilder<> BB(Body);
          Value *WP = BB.CreateInBoundsGEP(WTy, Words,
                                           {BB.getInt32(0), BB.CreateLShr(I, 2)});
          Value *Sh = BB.CreateShl(BB.CreateAnd(I, 3), 3);
          BB.CreateStore(BB.CreateOr(BB.CreateLoad(BB.getInt32Ty(), WP),
                                     BB.CreateShl(BB.CreateZExt(Ch, BB.getInt32Ty()),
                                                  Sh)),
                         WP);
          I->addIncoming(BB.CreateAdd(I, BB.getInt32(1)), Body);
          BB.CreateBr(Loop);
          B.SetInsertPoint(CI);
          W.append({B.getInt32(4), I, B.getInt32(StrWords)});
          for (unsigned K = 0; K < StrWords; ++K)
            W.push_back(B.CreateLoad(
                B.getInt32Ty(),
                B.CreateInBoundsGEP(WTy, Words, {B.getInt32(0), B.getInt32(K)})));
          continue;
        }
        unsigned Tag = 3;
        if (T->isFloatingPointTy()) {
          Tag = 2;
          A = B.CreateBitCast(B.CreateFPExt(A, B.getDoubleTy()), B.getInt64Ty());
        } else if (T->isIntegerTy()) {
          Tag = T->getIntegerBitWidth() > 32 ? 5 : 1;
          A = B.CreateSExtOrTrunc(A, B.getInt64Ty());
        } else {
          // Pointers into buffers have no address the host could print.
          A = B.getInt64(0);
        }
        W.append({B.getInt32(Tag), B.CreateTrunc(A, B.getInt32Ty()),
                  B.CreateTrunc(B.CreateLShr(A, 32), B.getInt32Ty())});
      }
      W[0] = B.getInt32(W.size());
      Value *Base = B.CreateAtomicRMW(
          AtomicRMWInst::Add, Buf, B.getInt32(W.size()), Align(4),
          AtomicOrdering::Monotonic, M.getContext().getOrInsertSyncScopeID("device"));
      for (unsigned K = 0; K < W.size(); ++K) {
        Value *Idx = B.CreateZExt(B.CreateAdd(Base, B.getInt32(2 + K)),
                                  B.getInt64Ty());
        Idx = B.CreateBinaryIntrinsic(Intrinsic::umin, Idx, B.getInt64(Last));
        B.CreateStore(W[K], B.CreateInBoundsGEP(BufTy, Buf, {B.getInt64(0), Idx}));
      }
      // chipStar's printf returns the number of arguments after the format.
      CI->replaceAllUsesWith(
          ConstantInt::get(CI->getType(), CI->arg_size() - 1));
      CI->eraseFromParent();
      Changed = true;
    }
    // Local arrays of string pointers left only written: Vulkan has no
    // pointers to keep in them.
    SmallVector<AllocaInst *> Allocas;
    for (Function &F : M)
      for (Instruction &I : instructions(F))
        if (auto *AI = dyn_cast<AllocaInst>(&I))
          Allocas.push_back(AI);
    for (AllocaInst *AI : Allocas)
      Changed |= eraseIfWriteOnly(AI);
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Defines each extern __shared__ array as one ChipVulkanDynSharedName array,
// which the runtime sizes per launch.
class HipVulkanDefineDynamicSharedPass
    : public HipPassInfoMixin<HipVulkanDefineDynamicSharedPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool Changed = false;
    GlobalVariable *NewGV = nullptr;
    for (GlobalVariable &GV : make_early_inc_range(M.globals())) {
      auto *Ty = dyn_cast<ArrayType>(GV.getValueType());
      if (!GV.isDeclaration() || !Ty ||
          GV.getAddressSpace() != SPIRV_WORKGROUP_AS)
        continue;
      // All extern __shared__ arrays start at the same address.
      if (!NewGV) {
        auto *ArrTy = ArrayType::get(Type::getInt8Ty(M.getContext()),
                                     ChipVulkanDynSharedBytes);
        NewGV = new GlobalVariable(
            M, ArrTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
            PoisonValue::get(ArrTy), ChipVulkanDynSharedName, &GV,
            GlobalValue::NotThreadLocal, SPIRV_WORKGROUP_AS);
      }
      NewGV->setAlignment(std::max(NewGV->getAlign().valueOrOne(),
                                   GV.getAlign().valueOrOne()));
      std::string Name = GV.getName().str();
      GV.replaceAllUsesWith(NewGV);
      GV.eraseFromParent();
      // As HipDynMem reports for OpenCL: the address has no constant form.
      for (GlobalVariable &Other : M.globals())
        if (Other.hasInitializer() &&
            refersTo(Other.getInitializer(), NewGV))
          report_fatal_error(
              "HipDynMem: a static or global variable is initialized with the "
              "address of dynamic shared memory '" +
                  Twine(Name) + "', which is unsupported",
              /*GenCrashDiag=*/false);
      Changed = true;
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

private:
  static bool refersTo(const Constant *C, const GlobalValue *GV) {
    return C == GV || any_of(C->operands(), [&](const Use &Op) {
             return isa<Constant>(Op) && refersTo(cast<Constant>(Op), GV);
           });
  }
};

// Moves read-only global and constant tables (math library coefficients,
// local array initializers) to addrspace(0), where the SPIR-V backend gives
// each function its own constant copy; Vulkan has no address space for them.
class HipVulkanPrivatizeConstantsPass
    : public HipPassInfoMixin<HipVulkanPrivatizeConstantsPass> {
  // Points Use at New, the addrspace(0) version of what it now points at.
  static bool rewrite(Use &U, Value *NewV) {
    User *Usr = U.getUser();
    if (auto *CE = dyn_cast<ConstantExpr>(Usr)) {
      auto *New = cast<Constant>(NewV);
      Constant *Repl = nullptr;
      if (CE->getOpcode() == Instruction::AddrSpaceCast)
        Repl = ConstantExpr::getAddrSpaceCast(New, CE->getType());
      else if (auto *GEP = dyn_cast<GEPOperator>(CE)) {
        SmallVector<Constant *, 4> Idx;
        for (Use &I : GEP->indices())
          Idx.push_back(cast<Constant>(I));
        Repl = ConstantExpr::getGetElementPtr(GEP->getSourceElementType(), New,
                                              Idx, GEP->getNoWrapFlags());
        // Rewrite the GEP's own users against the new GEP.
        for (Use &UU : make_early_inc_range(CE->uses()))
          if (!rewrite(UU, Repl))
            return false;
        return true;
      }
      if (!Repl)
        return false;
      CE->replaceAllUsesWith(Repl);
      return true;
    }
    if (auto *LI = dyn_cast<LoadInst>(Usr)) {
      LI->setOperand(0, NewV);
      return true;
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Usr)) {
      IRBuilder<> B(GEP);
      SmallVector<Value *, 4> Idx(GEP->indices());
      auto *NewGEP = cast<GetElementPtrInst>(B.Insert(GetElementPtrInst::Create(
          GEP->getSourceElementType(), NewV, Idx)));
      NewGEP->setNoWrapFlags(GEP->getNoWrapFlags());
      for (Use &UU : make_early_inc_range(GEP->uses()))
        if (!rewrite(UU, NewGEP))
          return false;
      GEP->eraseFromParent();
      return true;
    }
    if (auto *ASC = dyn_cast<AddrSpaceCastInst>(Usr)) {
      ASC->replaceAllUsesWith(
          IRBuilder<>(ASC).CreateAddrSpaceCast(NewV, ASC->getType()));
      ASC->eraseFromParent();
      return true;
    }
    return false;
  }

public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool Changed = false;
    for (GlobalVariable &GV : make_early_inc_range(M.globals())) {
      unsigned AS = GV.getAddressSpace();
      if ((AS != 1 && AS != 2) || !GV.isConstant() || !GV.hasInitializer())
        continue;
      auto *New = new GlobalVariable(
          M, GV.getValueType(), /*isConstant=*/true,
          GlobalValue::PrivateLinkage, GV.getInitializer(), GV.getName(),
          &GV, GlobalValue::NotThreadLocal, /*AddressSpace=*/0);
      New->setAlignment(GV.getAlign());
      bool Ok = true;
      for (Use &U : make_early_inc_range(GV.uses()))
        Ok &= rewrite(U, New);
      GV.removeDeadConstantUsers();
      if (Ok && GV.use_empty())
        GV.eraseFromParent();
      if (New->use_empty())
        New->eraseFromParent();
      Changed = true;
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Vulkan has no sized memory copies, so memcpy and memmove become loops;
// prefetches, mere hints, are dropped.
class HipVulkanExpandMemTransfersPass
    : public HipPassInfoMixin<HipVulkanExpandMemTransfersPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    SmallVector<IntrinsicInst *, 8> Work;
    for (Instruction &I : instructions(F))
      if (auto *II = dyn_cast<IntrinsicInst>(&I);
          II && (isa<MemTransferInst>(II) ||
                 II->getIntrinsicID() == Intrinsic::prefetch))
        Work.push_back(II);
    if (Work.empty())
      return PreservedAnalyses::all();
    auto &TTI = AM.getResult<TargetIRAnalysis>(F);
    for (IntrinsicInst *II : Work) {
      auto *Len = dyn_cast<ConstantInt>(II->getArgOperand(2));
      if (auto *Cpy = dyn_cast<MemCpyInst>(II);
          Cpy && Len && Len->getZExtValue() <= 256) {
        // Constant offsets, which a loop's byte index into a local struct is
        // not: SROA then resolves the copy.
        uint64_t N = Len->getZExtValue();
        uint64_t Chunk = std::min<uint64_t>(
            {8, Cpy->getSourceAlign().valueOrOne().value(),
             Cpy->getDestAlign().valueOrOne().value()});
        while (N % Chunk)
          Chunk /= 2;
        IRBuilder<> B(Cpy);
        Type *T = B.getIntNTy(Chunk * 8);
        for (uint64_t Off = 0; Off < N; Off += Chunk) {
          Value *Src =
              B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), Cpy->getSource(), Off);
          // Bytes of a constant global are constants: Vulkan cannot read it.
          Constant *Folded =
              isa<Constant>(Src) && !Cpy->isVolatile()
                  ? ConstantFoldLoadFromConstPtr(cast<Constant>(Src), T,
                                                 F.getDataLayout())
                  : nullptr;
          Value *V = Folded;
          if (!V)
            V = B.CreateAlignedLoad(T, Src, Align(Chunk), Cpy->isVolatile());
          B.CreateAlignedStore(
              V, B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), Cpy->getDest(), Off),
              Align(Chunk), Cpy->isVolatile());
        }
      } else if (auto *Cpy = dyn_cast<MemCpyInst>(II))
        expandMemCpyAsLoop(Cpy, TTI);
      else if (auto *Mov = dyn_cast<MemMoveInst>(II);
               Mov && !expandMemMoveAsLoop(Mov, TTI))
        continue;
      II->eraseFromParent();
    }
    return PreservedAnalyses::none();
  }
};

// Coerced pointer arguments arrive as inttoptr(ptrtoint p), possibly with
// offsets added in between, which hides p's address space from
// InferAddressSpaces; a cast of a byte GEP says the same.
class HipVulkanFoldPtrIntPairsPass
    : public HipPassInfoMixin<HipVulkanFoldPtrIntPairsPass> {
  // The one ptrtoint leaf of the add tree X, with the other leaves in Rest.
  static PtrToIntInst *splitBase(Value *X, SmallVectorImpl<Value *> &Rest) {
    if (auto *PTI = dyn_cast<PtrToIntInst>(X))
      return PTI;
    auto *BO = dyn_cast<BinaryOperator>(X);
    SmallVector<Value *, 4> R0, R1;
    PtrToIntInst *A = nullptr, *B = nullptr;
    if (BO && BO->getOpcode() == Instruction::Add) {
      A = splitBase(BO->getOperand(0), R0);
      B = splitBase(BO->getOperand(1), R1);
    }
    if (!A == !B) {
      Rest.push_back(X);
      return nullptr;
    }
    Rest.append(R0);
    Rest.append(R1);
    return A ? A : B;
  }

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    const DataLayout &DL = F.getDataLayout();
    bool Changed = false;
    for (Instruction &I : make_early_inc_range(instructions(F))) {
      auto *ITP = dyn_cast<IntToPtrInst>(&I);
      SmallVector<Value *, 4> Rest;
      auto *PTI = ITP ? splitBase(ITP->getOperand(0), Rest) : nullptr;
      if (!PTI ||
          DL.getPointerTypeSizeInBits(PTI->getPointerOperand()->getType()) !=
              PTI->getType()->getIntegerBitWidth() ||
          DL.getPointerTypeSizeInBits(ITP->getType()) !=
              PTI->getType()->getIntegerBitWidth())
        continue;
      IRBuilder<> B(ITP);
      Value *P = PTI->getPointerOperand();
      if (!Rest.empty()) {
        Value *Off = Rest[0];
        for (Value *R : drop_begin(Rest))
          Off = B.CreateAdd(Off, R);
        P = B.CreateGEP(B.getInt8Ty(), P, Off);
      }
      ITP->replaceAllUsesWith(
          B.CreatePointerBitCastOrAddrSpaceCast(P, ITP->getType()));
      ITP->eraseFromParent();
      Changed = true;
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// A pointer known to be global is not local and vice versa: to_local and
// to_global of it, casts between the two address spaces, are null. Folds the
// branches on them, whose other side Vulkan could not express.
class HipVulkanResolveAddrSpaceCastsPass
    : public HipPassInfoMixin<HipVulkanResolveAddrSpaceCastsPass> {
  static bool isNullCast(Value *V) {
    auto *ASC = dyn_cast<AddrSpaceCastOperator>(V);
    if (!ASC)
      return false;
    unsigned From = ASC->getSrcAddressSpace(), To = ASC->getDestAddressSpace();
    return isa<ConstantPointerNull>(ASC->getPointerOperand()) ||
           (From != To && From != SPIRV_GENERIC_AS && To != SPIRV_GENERIC_AS);
  }

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    bool Changed = false;
    const DataLayout &DL0 = F.getDataLayout();
    // A shared array has no address in Vulkan; its byte offset stands in.
    for (Instruction &I : make_early_inc_range(instructions(F)))
      if (auto *PTI = dyn_cast<PtrToIntInst>(&I)) {
        APInt Off(64, 0);
        auto *G = dyn_cast<GlobalVariable>(
            PTI->getPointerOperand()->stripAndAccumulateConstantOffsets(
                DL0, Off, true));
        if (G && G->getAddressSpace() == SPIRV_WORKGROUP_AS) {
          // Distinct objects get distinct offsets: their place in one layout.
          uint64_t Base = 0;
          for (GlobalVariable &O : F.getParent()->globals())
            if (O.getAddressSpace() == SPIRV_WORKGROUP_AS) {
              Base = alignTo(Base, O.getAlign().valueOrOne());
              if (&O == G)
                break;
              Base += DL0.getTypeAllocSize(O.getValueType());
            }
          PTI->replaceAllUsesWith(ConstantInt::get(PTI->getType(), Off + Base));
          PTI->eraseFromParent();
          Changed = true;
        }
      }
    for (Instruction &I : instructions(F))
      for (Use &U : I.operands())
        if (U->getType()->isPointerTy() && isNullCast(U)) {
          U.set(ConstantPointerNull::get(cast<PointerType>(U->getType())));
          Changed = true;
        }
    if (!Changed)
      return PreservedAnalyses::all();
    const DataLayout &DL = DL0;
    for (Instruction &I : make_early_inc_range(instructions(F)))
      if (isa<CmpInst>(I))
        if (Value *V = simplifyInstruction(&I, DL)) {
          I.replaceAllUsesWith(V);
          I.eraseFromParent();
        }
    for (BasicBlock &BB : F)
      ConstantFoldTerminator(&BB, /*DeleteDeadConditions=*/true);
    removeUnreachableBlocks(F);
    return PreservedAnalyses::none();
  }
};

// Vulkan has no pointers in push constants: each pointer field of a kernel's
// by-value argument becomes a trailing pointer argument named
// __chip_argfield_<argument number>_<byte offset>, stored over the field in a
// local copy, and the argument's type has integers in place of pointers.
class HipVulkanSplitPointerFieldsPass
    : public HipPassInfoMixin<HipVulkanSplitPointerFieldsPass> {
  // T with its pointers replaced by integers; collects their byte offsets.
  static Type *withoutPtrs(const DataLayout &DL, Type *T, uint64_t Off,
                           SmallVectorImpl<std::pair<uint64_t, Type *>> &PtrOffs) {
    if (T->isPointerTy()) {
      PtrOffs.emplace_back(Off, T);
      return DL.getIntPtrType(T);
    }
    if (auto *ST = dyn_cast<StructType>(T)) {
      const StructLayout *SL = DL.getStructLayout(ST);
      SmallVector<Type *, 8> Elems;
      for (auto [I, E] : enumerate(ST->elements()))
        Elems.push_back(
            withoutPtrs(DL, E, Off + SL->getElementOffset(I), PtrOffs));
      return StructType::get(T->getContext(), Elems, ST->isPacked());
    }
    if (auto *AT = dyn_cast<ArrayType>(T)) {
      Type *E = AT->getElementType();
      for (uint64_t I = 0; I < AT->getNumElements(); ++I)
        withoutPtrs(DL, E, Off + I * DL.getTypeAllocSize(E), PtrOffs);
      SmallVector<std::pair<uint64_t, Type *>, 4> Ignored;
      return ArrayType::get(withoutPtrs(DL, E, 0, Ignored), AT->getNumElements());
    }
    return T;
  }

public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    const DataLayout &DL = M.getDataLayout();
    LLVMContext &C = M.getContext();
    bool Changed = false;
    SmallVector<Function *, 8> Kernels;
    for (Function &F : M)
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL && !F.isDeclaration())
        Kernels.push_back(&F);
    for (Function *F : Kernels) {
      // (argument, pointer field offsets, type without pointers)
      SmallVector<
          std::tuple<unsigned, SmallVector<std::pair<uint64_t, Type *>, 4>,
                     Type *>,
          4>
          Split;
      for (Argument &A : F->args()) {
        if (!A.hasByValAttr())
          continue;
        SmallVector<std::pair<uint64_t, Type *>, 4> Offs;
        Type *NewT = withoutPtrs(DL, A.getParamByValType(), 0, Offs);
        if (!Offs.empty())
          Split.emplace_back(A.getArgNo(), Offs, NewT);
      }
      if (Split.empty())
        continue;
      Changed = true;
      SmallVector<Type *, 8> Params(F->getFunctionType()->params());
      SmallVector<std::string, 8> Names;
      for (auto &[ArgNo, Offs, NewT] : Split)
        for (uint64_t Off : make_first_range(Offs)) {
          Params.push_back(PointerType::get(C, SPIRV_CROSSWORKGROUP_AS));
          Names.push_back("__chip_argfield_" + std::to_string(ArgNo) + "_" +
                          std::to_string(Off));
        }
      auto *NF = Function::Create(
          FunctionType::get(F->getReturnType(), Params, false),
          F->getLinkage(), F->getAddressSpace(), "", &M);
      NF->copyAttributesFrom(F);
      NF->copyMetadata(F, 0);
      NF->takeName(F);
      NF->splice(NF->begin(), F);
      for (Argument &A : F->args()) {
        A.replaceAllUsesWith(NF->getArg(A.getArgNo()));
        NF->getArg(A.getArgNo())->takeName(&A);
      }
      unsigned Extra = F->arg_size();
      IRBuilder<> B(&*NF->getEntryBlock().getFirstInsertionPt());
      for (auto &[ArgNo, Offs, NewT] : Split) {
        Argument *A = NF->getArg(ArgNo);
        Type *T = A->getParamByValType();
        AllocaInst *Copy = B.CreateAlloca(T, DL.getAllocaAddrSpace());
        Copy->setAlignment(A->getParamAlign().valueOrOne());
        A->replaceAllUsesWith(Copy);
        B.CreateStore(B.CreateLoad(NewT, A), Copy);
        for (auto [Off, PtrTy] : Offs) {
          Argument *P = NF->getArg(Extra++);
          P->setName(Names[P->getArgNo() - F->arg_size()]);
          Value *Field = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), Copy, Off);
          B.CreateStore(B.CreatePointerBitCastOrAddrSpaceCast(P, PtrTy), Field);
        }
        NF->removeParamAttr(ArgNo, Attribute::ByVal);
        NF->addParamAttr(ArgNo, Attribute::getWithByValType(C, NewT));
      }
      F->replaceAllUsesWith(NF);
      F->eraseFromParent();
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Removes the hidden pointer arguments that optimization left unused: without
// a storage buffer the runtime could not tell them from client arguments.
class HipVulkanDropDeadHiddenArgsPass
    : public HipPassInfoMixin<HipVulkanDropDeadHiddenArgsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool Changed = false;
    for (Function &F : make_early_inc_range(M)) {
      if (F.getCallingConv() != CallingConv::SPIR_KERNEL || F.isDeclaration())
        continue;
      SmallVector<Type *, 8> Params;
      SmallVector<AttributeSet, 8> ParamAttrs;
      SmallVector<Argument *, 8> Kept;
      for (Argument &A : F.args())
        if (!A.use_empty() || !(A.getName().starts_with(ChipArgFieldPrefix) ||
                                A.getName().starts_with(ChipDevGlobalArgPrefix))) {
          Params.push_back(A.getType());
          ParamAttrs.push_back(F.getAttributes().getParamAttrs(A.getArgNo()));
          Kept.push_back(&A);
        }
      if (Kept.size() == F.arg_size())
        continue;
      Changed = true;
      auto *NF = Function::Create(
          FunctionType::get(F.getReturnType(), Params, false), F.getLinkage(),
          F.getAddressSpace(), "", &M);
      NF->copyMetadata(&F, 0);
      NF->setCallingConv(F.getCallingConv());
      NF->setAttributes(AttributeList::get(F.getContext(),
                                           F.getAttributes().getFnAttrs(),
                                           F.getAttributes().getRetAttrs(),
                                           ParamAttrs));
      NF->takeName(&F);
      NF->splice(NF->begin(), &F);
      for (auto [A, NA] : zip(Kept, NF->args())) {
        A->replaceAllUsesWith(&NA);
        NA.takeName(A);
      }
      F.replaceAllUsesWith(NF);
      F.eraseFromParent();
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Vulkan has no program-scope storage, so static and function-local
// __device__ variables, which the host does not register, become device
// variables like registered ones: the runtime binds those it finds through
// their shadow kernels.
class HipVulkanExposeStaticGlobalsPass
    : public HipPassInfoMixin<HipVulkanExposeStaticGlobalsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool Changed = false;
    for (GlobalVariable &GV : M.globals())
      if (GV.getAddressSpace() == SPIRV_CROSSWORKGROUP_AS && !GV.isConstant() &&
          GV.hasInitializer() && !GV.hasSection() && GV.hasName() &&
          !GV.use_empty() &&
          (!GV.isExternallyInitialized() || GV.hasComdat())) {
        // An inline variable's COMDAT makes HipGlobalVariables treat it as
        // unregistered; nothing is linked after this anyway.
        GV.setComdat(nullptr);
        GV.setExternallyInitialized(true);
        GV.setUnnamedAddr(GlobalValue::UnnamedAddr::None);
        Changed = true;
      }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Private arrays become one byte array, each at its own offset; returns it.
static AllocaInst *mergeAllocas(Function &F, ArrayRef<AllocaInst *> Members,
                                const DataLayout &DL) {
  uint64_t Size = 0;
  Align MaxAlign;
  SmallVector<uint64_t, 4> Offsets;
  for (AllocaInst *A : Members) {
    Size = alignTo(Size, A->getAlign());
    Offsets.push_back(Size);
    Size += DL.getTypeAllocSize(A->getAllocatedType());
    MaxAlign = std::max(MaxAlign, A->getAlign());
  }
  IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
  auto *Merged = B.CreateAlloca(ArrayType::get(B.getInt8Ty(), Size),
                                Members[0]->getAddressSpace());
  Merged->setAlignment(MaxAlign);
  // All replacements first: the builder may sit at one of the members.
  SmallVector<Value *, 4> Parts;
  for (uint64_t Off : Offsets)
    Parts.push_back(B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), Merged, Off));
  for (auto [A, Part] : zip(Members, Parts)) {
    for (User *U : make_early_inc_range(A->users()))
      if (auto *II = dyn_cast<IntrinsicInst>(U); II && II->isLifetimeStartOrEnd())
        II->eraseFromParent();
    A->replaceAllUsesWith(Part);
    A->eraseFromParent();
  }
  return Merged;
}

// A private array of pointers to private arrays, a table indexed at run time,
// has no Vulkan form: Vulkan memory cannot hold pointers. The arrays become
// one and the table holds byte offsets into it.
class HipVulkanPointerTablesPass
    : public HipPassInfoMixin<HipVulkanPointerTablesPass> {
  // Collects the pointer loads and stores through V; false on other uses.
  static bool collect(Value *V, SmallVectorImpl<Instruction *> &Acc) {
    for (User *U : V->users()) {
      if (isa<GetElementPtrInst, AddrSpaceCastInst>(U) &&
          U->getOperand(0) == V) {
        if (!collect(U, Acc))
          return false;
      } else if (auto *L = dyn_cast<LoadInst>(U);
                 L && L->getType()->isPointerTy()) {
        Acc.push_back(L);
      } else if (auto *S = dyn_cast<StoreInst>(U);
                 S && S->getPointerOperand() == V &&
                 S->getValueOperand()->getType()->isPointerTy()) {
        Acc.push_back(S);
      } else if (auto *II = dyn_cast<IntrinsicInst>(U);
                 !II || !II->isLifetimeStartOrEnd()) {
        return false;
      }
    }
    return true;
  }
  // The array P points into through casts and GEPs, or null.
  static AllocaInst *arrayOf(Value *P) {
    while (true) {
      if (auto *A = dyn_cast<AllocaInst>(P))
        return isa<ArrayType>(A->getAllocatedType()) ? A : nullptr;
      if (auto *G = dyn_cast<GEPOperator>(P))
        P = G->getPointerOperand();
      else if (auto *C = dyn_cast<AddrSpaceCastOperator>(P))
        P = C->getPointerOperand();
      else
        return nullptr;
    }
  }
  // The byte offset of P from Base, which arrayOf found, emitted at B.
  static Value *offsetFrom(Value *P, Value *Base, IRBuilder<> &B,
                           const DataLayout &DL) {
    Value *Off = B.getInt64(0);
    while (P != Base) {
      if (auto *G = dyn_cast<GEPOperator>(P)) {
        Off = B.CreateAdd(Off, B.CreateSExtOrTrunc(
                                   emitGEPOffset(&B, DL, cast<User>(G)),
                                   B.getInt64Ty()));
        P = G->getPointerOperand();
      } else {
        P = cast<AddrSpaceCastOperator>(P)->getPointerOperand();
      }
    }
    return Off;
  }

  // A table whose slots are each written once at a constant index before
  // any read: each read becomes a select among the stored pointers.
  static bool selectTable(AllocaInst *T, ArrayRef<Instruction *> Acc,
                          DominatorTree &DT, const DataLayout &DL) {
    unsigned N = T->getAllocatedType()->getArrayNumElements();
    SmallVector<Value *, 8> Slot(N, nullptr);
    SmallVector<StoreInst *, 8> Stores;
    for (Instruction *I : Acc)
      if (auto *S = dyn_cast<StoreInst>(I)) {
        APInt Off(64, 0);
        if (S->getPointerOperand()->stripAndAccumulateConstantOffsets(
                DL, Off, true) != T ||
            Off.getZExtValue() % 8 || Off.getZExtValue() / 8 >= N ||
            Slot[Off.getZExtValue() / 8])
          return false;
        Slot[Off.getZExtValue() / 8] = S->getValueOperand();
        Stores.push_back(S);
      }
    for (Instruction *I : Acc)
      if (auto *L = dyn_cast<LoadInst>(I))
        for (StoreInst *S : Stores)
          if (!DT.dominates(S, L))
            return false;
    for (Instruction *I : Acc) {
      auto *L = dyn_cast<LoadInst>(I);
      if (!L)
        continue;
      IRBuilder<> B(L);
      Value *Idx =
          B.CreateLShr(offsetFrom(L->getPointerOperand(), T, B, DL), 3);
      Value *V = PoisonValue::get(L->getType());
      for (unsigned K = N; K-- > 0;)
        if (Slot[K])
          V = B.CreateSelect(B.CreateICmpEQ(Idx, B.getInt64(K)),
                             B.CreatePointerBitCastOrAddrSpaceCast(
                                 Slot[K], L->getType()),
                             V);
      L->replaceAllUsesWith(V);
    }
    for (Instruction *I : Acc)
      I->eraseFromParent();
    // Only address computations and lifetime markers are left.
    SmallVector<Instruction *, 8> Dead, Work{T};
    while (!Work.empty())
      for (User *U : Work.pop_back_val()->users()) {
        Dead.push_back(cast<Instruction>(U));
        Work.push_back(cast<Instruction>(U));
      }
    for (Instruction *I : reverse(Dead))
      I->eraseFromParent();
    T->eraseFromParent();
    return true;
  }

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    const DataLayout &DL = F.getDataLayout();
    SmallVector<WeakVH, 4> Tables;
    for (Instruction &I : instructions(F))
      if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        auto *AT = dyn_cast<ArrayType>(AI->getAllocatedType());
        if (AT && AT->getElementType()->isPointerTy() &&
            DL.getTypeAllocSize(AT->getElementType()) == 8)
          Tables.push_back(AI);
      }
    bool Changed = false;
    for (WeakVH &VH : Tables) {
      // Null once merged into another table's arrays.
      auto *T = cast_or_null<AllocaInst>(VH);
      if (!T)
        continue;
      SmallVector<Instruction *, 8> Acc;
      SmallSetVector<AllocaInst *, 4> Arrays;
      if (!collect(T, Acc))
        continue;
      if (!all_of(Acc, [&](Instruction *I) {
            auto *S = dyn_cast<StoreInst>(I);
            AllocaInst *A = S ? arrayOf(S->getValueOperand()) : T;
            if (A && A != T)
              Arrays.insert(A);
            return A && (A != T || !S);
          }) || Arrays.empty()) {
        Changed |= selectTable(T, Acc, AM.getResult<DominatorTreeAnalysis>(F),
                               DL);
        continue;
      }
      AllocaInst *M = Arrays.size() == 1
                          ? Arrays[0]
                          : mergeAllocas(F, Arrays.getArrayRef(), DL);
      for (Instruction *I : Acc) {
        IRBuilder<> B(I);
        if (auto *S = dyn_cast<StoreInst>(I)) {
          B.CreateAlignedStore(offsetFrom(S->getValueOperand(), M, B, DL),
                               S->getPointerOperand(), S->getAlign(),
                               S->isVolatile());
        } else {
          auto *L = cast<LoadInst>(I);
          Value *Off = B.CreateAlignedLoad(B.getInt64Ty(),
                                           L->getPointerOperand(),
                                           L->getAlign(), L->isVolatile());
          L->replaceAllUsesWith(B.CreatePointerBitCastOrAddrSpaceCast(
              B.CreateGEP(B.getInt8Ty(), M, Off), L->getType()));
        }
        I->eraseFromParent();
      }
      IRBuilder<> B(T);
      auto *NT = B.CreateAlloca(
          ArrayType::get(B.getInt64Ty(),
                         T->getAllocatedType()->getArrayNumElements()),
          T->getAddressSpace());
      NT->setAlignment(T->getAlign());
      NT->takeName(T);
      T->replaceAllUsesWith(NT);
      T->eraseFromParent();
      Changed = true;
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Vulkan cannot reinterpret an array or index it by bytes: a shared or private
// array accessed only as some type T becomes an array of T, each access
// indexing it by its byte offset (a char buffer used as unsigned short, say).
class HipVulkanRetypeArraysPass
    : public HipPassInfoMixin<HipVulkanRetypeArraysPass> {
  static unsigned pointerIndex(Instruction *I) {
    return isa<StoreInst>(I) ? StoreInst::getPointerOperandIndex() : 0;
  }
  static Type *accessType(Instruction *I) {
    if (auto *CX = dyn_cast<AtomicCmpXchgInst>(I))
      return CX->getNewValOperand()->getType();
    return isa<AtomicRMWInst>(I) ? I->getType() : getLoadStoreType(I);
  }
  // Collects the accesses through V and its lifetime markers, and the pointers
  // derived from it in Derived; false if it has other uses.
  static bool collect(Value *V, SmallVectorImpl<Instruction *> &Accesses,
                      SmallVectorImpl<Instruction *> &Markers,
                      SetVector<Value *> &Derived) {
    if (auto *C = dyn_cast<Constant>(V))
      C->removeDeadConstantUsers();
    for (Use &U : V->uses()) {
      User *Usr = U.getUser();
      if (auto *I = dyn_cast<Instruction>(Usr); I && isInstructionTriviallyDead(I))
        continue;
      if (isa<PHINode>(Usr) ||
          (isa<SelectInst>(Usr) && U.getOperandNo() != 0) ||
          (isa<GEPOperator, AddrSpaceCastOperator>(Usr) &&
           U.getOperandNo() == 0)) {
        if (Derived.insert(Usr) && !collect(Usr, Accesses, Markers, Derived))
          return false;
      } else if (isa<LoadInst, StoreInst, AtomicRMWInst, AtomicCmpXchgInst>(
                     Usr) &&
                 U.getOperandNo() == pointerIndex(cast<Instruction>(Usr))) {
        Accesses.push_back(cast<Instruction>(Usr));
      } else if (auto *II = dyn_cast<IntrinsicInst>(Usr);
                 II && II->isLifetimeStartOrEnd()) {
        Markers.push_back(II);
      } else {
        return false;
      }
    }
    return true;
  }
  // As collect, and also false if a phi or select takes a pointer that does
  // not come from Obj.
  static bool collectAll(Value *Obj, SmallVectorImpl<Instruction *> &Accesses,
                         SmallVectorImpl<Instruction *> &Markers,
                         SetVector<Value *> &Derived) {
    if (!collect(Obj, Accesses, Markers, Derived))
      return false;
    for (Value *D : Derived)
      if (isa<PHINode, SelectInst>(D))
        for (Value *W : cast<User>(D)->operands())
          if (W->getType()->isPointerTy() && W != Obj &&
              !isa<PoisonValue, UndefValue>(W) && !Derived.count(W))
            return false;
    return true;
  }
  // The byte offset of P from Obj, emitted next to P's definitions.
  static Value *byteOffset(Value *P, Value *Obj, const DataLayout &DL,
                           DenseMap<Value *, Value *> &Memo) {
    Type *I64 = Type::getInt64Ty(Obj->getContext());
    if (P == Obj || isa<PoisonValue, UndefValue>(P))
      return ConstantInt::get(I64, 0);
    if (Value *O = Memo.lookup(P))
      return O;
    if (auto *Phi = dyn_cast<PHINode>(P)) {
      auto *O = PHINode::Create(I64, Phi->getNumIncomingValues(), "",
                                Phi->getIterator());
      Memo[P] = O;
      for (auto [V, BB] : zip(Phi->incoming_values(), Phi->blocks()))
        O->addIncoming(byteOffset(V, Obj, DL, Memo), BB);
      return O;
    }
    // Constant pointers fold to constant offsets.
    IRBuilder<> B(P->getContext());
    if (auto *I = dyn_cast<Instruction>(P))
      B.SetInsertPoint(I);
    if (auto *Sel = dyn_cast<SelectInst>(P))
      return Memo[P] = B.CreateSelect(
                 Sel->getCondition(),
                 byteOffset(Sel->getTrueValue(), Obj, DL, Memo),
                 byteOffset(Sel->getFalseValue(), Obj, DL, Memo));
    Value *Base = byteOffset(cast<Operator>(P)->getOperand(0), Obj, DL, Memo);
    auto *G = dyn_cast<GEPOperator>(P);
    if (!G)
      return Memo[P] = Base;
    APInt C(64, 0);
    Value *O = G->accumulateConstantOffset(DL, C)
                   ? ConstantInt::get(I64, C.getSExtValue())
                   : emitGEPOffset(&B, DL, cast<User>(G));
    return Memo[P] = B.CreateAdd(Base, B.CreateSExtOrTrunc(O, I64));
  }
  // The narrowest type Is access, if the others are plain loads and stores of
  // a whole number of it (which are split) and Total bytes are too.
  static Type *uniformType(ArrayRef<Instruction *> Is, uint64_t Total,
                           const DataLayout &DL) {
    Type *T = accessType(Is[0]);
    for (Instruction *I : Is)
      if (DL.getTypeAllocSize(accessType(I)) < DL.getTypeAllocSize(T))
        T = accessType(I);
    uint64_t Size = DL.getTypeAllocSize(T);
    if (!isPowerOf2_64(Size) || Total % Size || T->isAggregateType())
      return nullptr;
    for (Instruction *I : Is) {
      // The index drops the offset's low bits: each must be aligned to T.
      if (isa<LoadInst, StoreInst>(I) && getLoadStoreAlignment(I).value() < Size)
        return nullptr;
      Type *A = accessType(I);
      uint64_t ASize = DL.getTypeAllocSize(A);
      auto *L = dyn_cast<LoadInst>(I);
      auto *S = dyn_cast<StoreInst>(I);
      if (A != T && (!(L ? L->isSimple() : S && S->isSimple()) ||
                     A->isAggregateType() ||
                     A->isPtrOrPtrVectorTy() || T->isPtrOrPtrVectorTy() ||
                     ASize % Size || DL.getTypeSizeInBits(A) != ASize * 8))
        return nullptr;
    }
    return T;
  }
  // Points the accesses Is of Obj at NewObj, an array of their type T.
  static void rewrite(Value *Obj, Value *NewObj, ArrayType *ArrTy,
                      ArrayRef<Instruction *> Is, const DataLayout &DL,
                      DenseMap<Value *, Value *> &Memo) {
    Type *T = ArrTy->getElementType();
    uint64_t Size = DL.getTypeAllocSize(T);
    for (Instruction *I : Is) {
      Value *Off = byteOffset(I->getOperand(pointerIndex(I)), Obj, DL, Memo);
      IRBuilder<> B(I);
      Value *Idx = B.CreateLShr(Off, Log2_64(Size));
      auto Elt = [&](unsigned K) {
        return B.CreateInBoundsGEP(
            ArrTy, NewObj, {B.getInt64(0), B.CreateAdd(Idx, B.getInt64(K))});
      };
      Type *A = accessType(I);
      if (A == T) {
        I->setOperand(pointerIndex(I), Elt(0));
        continue;
      }
      // A wider access: one per element, through an integer of its width.
      // WORKAROUND(CHIP-SPV/chipStar#1742, no upstream report): GlobalISel
      // asserts on a scalar to vector bitcast wider than a shader vector.
      // Remove the integer detour when CanaryVulkanBitcastFewerElements fires.
      unsigned N = DL.getTypeAllocSize(A) / Size;
      Type *WideTy = B.getIntNTy(N * Size * 8), *EltTy = B.getIntNTy(Size * 8);
      if (auto *St = dyn_cast<StoreInst>(I)) {
        Value *V = B.CreateBitCast(St->getValueOperand(), WideTy);
        for (unsigned K = 0; K < N; ++K)
          B.CreateAlignedStore(
              B.CreateBitCast(
                  B.CreateTrunc(B.CreateLShr(V, K * Size * 8), EltTy), T),
              Elt(K), commonAlignment(St->getAlign(), K * Size),
              St->isVolatile());
      } else {
        auto *Ld = cast<LoadInst>(I);
        Value *V = ConstantInt::get(WideTy, 0);
        for (unsigned K = 0; K < N; ++K) {
          Value *E = B.CreateAlignedLoad(
              T, Elt(K), commonAlignment(Ld->getAlign(), K * Size),
              Ld->isVolatile());
          V = B.CreateOr(V, B.CreateShl(B.CreateZExt(B.CreateBitCast(E, EltTy),
                                                     WideTy),
                                        K * Size * 8));
        }
        Ld->replaceAllUsesWith(B.CreateBitCast(V, A));
      }
      I->eraseFromParent();
    }
  }

  // Private arrays a pointer chooses between at run time become one byte
  // array, each at its own offset, so that the choice is between offsets.
  static bool mergeChosenAllocas(Function &F, const DataLayout &DL) {
    DenseMap<AllocaInst *, AllocaInst *> Parent;
    std::function<AllocaInst *(AllocaInst *)> Find = [&](AllocaInst *A) {
      AllocaInst *P = Parent.lookup(A);
      return !P || P == A ? A : Parent[A] = Find(P);
    };
    for (Instruction &I : instructions(F)) {
      if (!isa<PHINode, SelectInst>(I) || !I.getType()->isPointerTy())
        continue;
      AllocaInst *First = nullptr;
      for (Value *V : isa<PHINode>(I) ? I.operands() : drop_begin(I.operands()))
        if (auto *AI = dyn_cast<AllocaInst>(getUnderlyingObject(V));
            AI && isa<ArrayType>(AI->getAllocatedType())) {
          Parent.try_emplace(AI, AI);
          if (First && Find(First) != Find(AI))
            Parent[Find(AI)] = Find(First);
          First = First ? First : AI;
        }
    }
    MapVector<AllocaInst *, SmallVector<AllocaInst *, 4>> Groups;
    for (auto &[A, P] : Parent)
      Groups[Find(A)].push_back(A);
    bool Changed = false;
    for (auto &[Root, Members] : Groups) {
      if (Members.size() < 2)
        continue;
      mergeAllocas(F, Members, DL);
      Changed = true;
    }
    return Changed;
  }

  // The derived pointers are now used only among themselves.
  static void eraseDerived(SetVector<Value *> &Derived) {
    SmallVector<Instruction *, 8> Dead;
    for (Value *D : Derived)
      if (auto *I = dyn_cast<Instruction>(D)) {
        I->replaceAllUsesWith(PoisonValue::get(I->getType()));
        Dead.push_back(I);
      }
    for (Instruction *I : Dead)
      I->eraseFromParent();
  }

public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    const DataLayout &DL = M.getDataLayout();
    bool Changed = false;
    for (GlobalVariable &GV : make_early_inc_range(M.globals())) {
      if (GV.getAddressSpace() != SPIRV_WORKGROUP_AS || GV.isDeclaration() ||
          !isa<ArrayType>(GV.getValueType()))
        continue;
      SmallVector<Instruction *, 8> Accesses, Markers;
      SetVector<Value *> Derived;
      if (!collectAll(&GV, Accesses, Markers, Derived) || Accesses.empty())
        continue;
      // Each function gets its own array: the kernels of a module may use
      // their dynamic shared memory as different types.
      MapVector<Function *, SmallVector<Instruction *, 4>> ByFunction;
      for (Instruction *I : Accesses)
        ByFunction[I->getFunction()].push_back(I);
      uint64_t Total = DL.getTypeAllocSize(GV.getValueType());
      if (!all_of(make_second_range(ByFunction), [&](auto &Is) {
            return uniformType(Is, Total, DL);
          }))
        continue;
      DenseMap<Value *, Value *> Memo;
      for (auto &[F, Is] : ByFunction) {
        Type *T = uniformType(Is, Total, DL);
        auto *ArrTy = ArrayType::get(T, Total / DL.getTypeAllocSize(T));
        auto *NewGV = new GlobalVariable(
            M, ArrTy, false, GlobalValue::InternalLinkage,
            PoisonValue::get(ArrTy), GV.getName() + "." + F->getName(), &GV,
            GlobalValue::NotThreadLocal, SPIRV_WORKGROUP_AS);
        NewGV->setAlignment(
            std::max(GV.getAlign().valueOrOne(), DL.getABITypeAlign(T)));
        rewrite(&GV, NewGV, ArrTy, Is, DL, Memo);
      }
      eraseDerived(Derived);
      // Only dead address computations still use it.
      GV.replaceAllUsesWith(PoisonValue::get(GV.getType()));
      GV.eraseFromParent();
      Changed = true;
    }
    for (Function &F : M) {
      Changed |= mergeChosenAllocas(F, DL);
      SmallVector<AllocaInst *, 8> Allocas;
      for (Instruction &I : instructions(F))
        if (auto *AI = dyn_cast<AllocaInst>(&I);
            AI && isa<ArrayType>(AI->getAllocatedType()))
          Allocas.push_back(AI);
      for (AllocaInst *AI : Allocas) {
        SmallVector<Instruction *, 8> Accesses, Markers;
        SetVector<Value *> Derived;
        if (!collectAll(AI, Accesses, Markers, Derived) || Accesses.empty())
          continue;
        uint64_t Total = DL.getTypeAllocSize(AI->getAllocatedType());
        Type *T = uniformType(Accesses, Total, DL);
        if (!T)
          continue;
        auto *ArrTy = ArrayType::get(T, Total / DL.getTypeAllocSize(T));
        auto *NewAI = new AllocaInst(ArrTy, AI->getAddressSpace(), "",
                                     AI->getIterator());
        NewAI->setAlignment(std::max(AI->getAlign(), DL.getABITypeAlign(T)));
        NewAI->takeName(AI);
        DenseMap<Value *, Value *> Memo;
        rewrite(AI, NewAI, ArrTy, Accesses, DL, Memo);
        eraseDerived(Derived);
        for (Instruction *Marker : Markers)
          Marker->eraseFromParent();
        AI->replaceAllUsesWith(PoisonValue::get(AI->getType()));
        AI->eraseFromParent();
        Changed = true;
      }
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// A global pointer chosen at run time among buffer arguments (a basis matrix
// picked by a switch, say) has no Vulkan form: each buffer is its own binding.
// It becomes an argument number and a byte offset, and every access through
// it branches on the number.
class HipVulkanSplitBufferChoicesPass
    : public HipPassInfoMixin<HipVulkanSplitBufferChoicesPass> {
  // Adds the arguments V derives from to Args; false if it derives from
  // anything but arguments, null and poison through GEPs, phis and selects.
  static bool leaves(Value *V, SmallSetVector<Argument *, 4> &Args,
                     SmallPtrSetImpl<Value *> &Seen) {
    if (!Seen.insert(V).second)
      return true;
    if (auto *A = dyn_cast<Argument>(V))
      return Args.insert(A), true;
    if (isa<ConstantPointerNull, PoisonValue, UndefValue>(V))
      return true;
    if (auto *G = dyn_cast<GEPOperator>(V))
      return leaves(G->getPointerOperand(), Args, Seen);
    if (auto *Sel = dyn_cast<SelectInst>(V))
      return leaves(Sel->getTrueValue(), Args, Seen) &&
             leaves(Sel->getFalseValue(), Args, Seen);
    if (auto *Phi = dyn_cast<PHINode>(V))
      return all_of(Phi->incoming_values(),
                    [&](Value *In) { return leaves(In, Args, Seen); });
    return false;
  }

  struct Rewriter {
    ArrayRef<Argument *> Args;
    const DataLayout &DL;
    DenseMap<Value *, std::pair<Value *, Value *>> Memo{};
    // (argument number, byte offset) of V, emitted next to V.
    std::pair<Value *, Value *> get(Value *V) {
      LLVMContext &C = V->getContext();
      Type *I32 = Type::getInt32Ty(C), *I64 = Type::getInt64Ty(C);
      if (auto *A = dyn_cast<Argument>(V))
        return {ConstantInt::get(I32, find(Args, A) - Args.begin()),
                ConstantInt::get(I64, 0)};
      if (isa<Constant>(V))
        return {ConstantInt::get(I32, 0), ConstantInt::get(I64, 0)};
      if (auto It = Memo.find(V); It != Memo.end())
        return It->second;
      if (auto *Phi = dyn_cast<PHINode>(V)) {
        auto *N = PHINode::Create(I32, Phi->getNumIncomingValues(), "",
                                  Phi->getIterator());
        auto *O = PHINode::Create(I64, Phi->getNumIncomingValues(), "",
                                  Phi->getIterator());
        Memo[V] = {N, O};
        for (auto [In, BB] : zip(Phi->incoming_values(), Phi->blocks())) {
          auto [IN, IO] = get(In);
          N->addIncoming(IN, BB);
          O->addIncoming(IO, BB);
        }
        return {N, O};
      }
      IRBuilder<> B(cast<Instruction>(V));
      if (auto *Sel = dyn_cast<SelectInst>(V)) {
        auto [TN, TO] = get(Sel->getTrueValue());
        auto [FN, FO] = get(Sel->getFalseValue());
        return Memo[V] = {B.CreateSelect(Sel->getCondition(), TN, FN),
                          B.CreateSelect(Sel->getCondition(), TO, FO)};
      }
      auto *G = cast<GEPOperator>(V);
      auto [N, O] = get(G->getPointerOperand());
      return Memo[V] = {N, B.CreateAdd(O, B.CreateSExtOrTrunc(
                                              emitGEPOffset(&B, DL, G), I64))};
    }
  };

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    SmallVector<Instruction *, 8> Accesses;
    for (Instruction &I : instructions(F))
      if (isa<LoadInst, StoreInst, AtomicRMWInst, AtomicCmpXchgInst>(I))
        Accesses.push_back(&I);
    bool Changed = false;
    for (Instruction *A : Accesses) {
      unsigned PtrIdx = isa<StoreInst>(A) ? StoreInst::getPointerOperandIndex()
                                          : 0;
      Value *P = A->getOperand(PtrIdx);
      if (P->getType()->getPointerAddressSpace() != SPIRV_CROSSWORKGROUP_AS)
        continue;
      SmallSetVector<Argument *, 4> Args;
      SmallPtrSet<Value *, 16> Seen;
      if (!leaves(P, Args, Seen) || Args.size() < 2)
        continue;
      Rewriter R{Args.getArrayRef(), F.getDataLayout()};
      auto [Which, Off] = R.get(P);
      // if (Which == 0) ... else if (Which == 1) ... else <last argument>.
      std::function<Value *(unsigned, Instruction *)> Branch =
          [&](unsigned K, Instruction *Before) -> Value * {
        auto Emit = [&](Instruction *At) {
          Value *Ptr = IRBuilder<>(At).CreateGEP(
              Type::getInt8Ty(At->getContext()), Args[K], Off);
          Instruction *C = A->clone();
          C->insertBefore(At->getIterator());
          C->setOperand(PtrIdx, Ptr);
          return C;
        };
        if (K + 1 == Args.size())
          return Emit(Before);
        Instruction *Then, *Else;
        SplitBlockAndInsertIfThenElse(
            IRBuilder<>(Before).CreateICmpEQ(
                Which, ConstantInt::get(Which->getType(), K)),
            Before, &Then, &Else);
        Value *TV = Emit(Then);
        Value *EV = Branch(K + 1, Else);
        if (A->getType()->isVoidTy())
          return nullptr;
        PHINode *Res = PHINode::Create(A->getType(), 2, "", Before->getIterator());
        Res->addIncoming(TV, Then->getParent());
        Res->addIncoming(EV, Else->getParent());
        return Res;
      };
      if (Value *Res = Branch(0, A))
        A->replaceAllUsesWith(Res);
      A->eraseFromParent();
      Changed = true;
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// A generic pointer chosen at run time between global memory and a shared or
// private array (a workspace in either) has no Vulkan form. It becomes a
// global pointer, a local one and a flag, and every access through it
// branches on the flag.
class HipVulkanSplitMixedPointersPass
    : public HipPassInfoMixin<HipVulkanSplitMixedPointersPass> {
  // The non-generic pointer V casts, or null.
  static Value *classify(Value *V) {
    Value *X = V->stripPointerCasts();
    return X->getType()->getPointerAddressSpace() == SPIRV_GENERIC_AS ? nullptr
                                                                      : X;
  }

  // The pointer a load, store or atomic accesses, or null.
  static Value *accessPointer(Instruction *I) {
    if (isa<AtomicRMWInst, AtomicCmpXchgInst>(I))
      return I->getOperand(0);
    return getLoadStorePointerOperand(I);
  }

  // Splits P if it mixes address spaces; returns false otherwise.
  static bool split(Instruction *P) {
    SmallVector<Value *, 4> In;
    for (Value *V : isa<PHINode>(P) ? P->operands()
                                    : drop_begin(P->operands()))
      In.push_back(classify(V));
    if (is_contained(In, nullptr))
      return false;
    SmallVector<Type *, 3> Spaces;
    for (Value *X : In)
      if (!is_contained(Spaces, X->getType()))
        Spaces.push_back(X->getType());
    if (Spaces.size() < 2)
      return false;
    // A phi's other incoming values need stand-ins valid on every edge: an
    // argument a global pointer derives from, or poison for a local array.
    auto *Phi = dyn_cast<PHINode>(P);
    SmallVector<Value *, 3> StandIn;
    for (Type *T : Spaces) {
      Value *S = nullptr;
      for (Value *X : In)
        if (X->getType() == T) {
          Value *O = getUnderlyingObject(X);
          if (!Phi || isa<Argument, Constant>(X))
            S = X;
          else if (isa<Argument>(O) && O->getType() == T && !S)
            S = O;
        }
      if (!S && Phi && T->getPointerAddressSpace() != SPIRV_CROSSWORKGROUP_AS)
        S = PoisonValue::get(T);
      if (!S)
        return false;
      StandIn.push_back(S);
    }
    // Only GEPs and accesses use it.
    SmallVector<Instruction *, 8> Work{P}, Accesses;
    for (size_t I = 0; I < Work.size(); ++I)
      for (User *U : Work[I]->users()) {
        auto *UI = cast<Instruction>(U);
        if (isa<GetElementPtrInst>(UI) &&
            cast<GetElementPtrInst>(UI)->getPointerOperand() == Work[I])
          Work.push_back(UI);
        else if (accessPointer(UI) == Work[I])
          Accesses.push_back(UI);
        else
          return false;
      }
    IRBuilder<> B(P);
    auto Pick = [&](function_ref<Value *(Value *)> F) -> Value * {
      // One candidate needs no select or phi: logical SPIR-V forbids both.
      Value *X0 = F(In[0]);
      if (all_equal(map_range(In, F)) && (!Phi || isa<Constant, Argument>(X0)))
        return X0;
      if (!Phi)
        return B.CreateSelect(cast<SelectInst>(P)->getCondition(), F(In[0]),
                              F(In[1]));
      PHINode *N = B.CreatePHI(F(In[0])->getType(), In.size());
      for (auto [X, BB] : zip(In, Phi->blocks()))
        N->addIncoming(F(X), BB);
      return N;
    };
    auto IndexOf = [&](Value *X) {
      return (unsigned)(find(Spaces, X->getType()) - Spaces.begin());
    };
    SmallVector<Value *, 3> Ptrs;
    for (auto [K, T] : enumerate(Spaces))
      Ptrs.push_back(Pick([&, K = K](Value *X) {
        return IndexOf(X) == K ? X : StandIn[K];
      }));
    Value *Which = Pick([&](Value *X) -> Value * {
      return B.getInt32(IndexOf(X));
    });
    unsigned PtrIdx = 0;
    for (Instruction *A : Accesses) {
      // The access's pointer, rebuilt on each space's pointer.
      SmallVector<GetElementPtrInst *, 4> Chain;
      for (Value *V = accessPointer(A); V != P;
           V = cast<GetElementPtrInst>(V)->getPointerOperand())
        Chain.push_back(cast<GetElementPtrInst>(V));
      PtrIdx = isa<StoreInst>(A) ? StoreInst::getPointerOperandIndex() : 0;
      auto Emit = [&](unsigned K, Instruction *Before) {
        IRBuilder<> EB(Before);
        Value *Ptr = Ptrs[K];
        for (GetElementPtrInst *G : reverse(Chain)) {
          SmallVector<Value *, 4> Idx(G->indices());
          Ptr = EB.CreateGEP(G->getSourceElementType(), Ptr, Idx);
        }
        Instruction *C = A->clone();
        C->insertBefore(Before->getIterator());
        C->setOperand(PtrIdx, Ptr);
        return C;
      };
      // if (Which == 0) ... else if (Which == 1) ... else <last space>.
      std::function<Value *(unsigned, Instruction *)> Branch =
          [&](unsigned K, Instruction *Before) -> Value * {
        if (K + 1 == Spaces.size())
          return Emit(K, Before);
        Instruction *Then, *Else;
        SplitBlockAndInsertIfThenElse(
            IRBuilder<>(Before).CreateICmpEQ(Which,
                                             ConstantInt::get(Which->getType(), K)),
            Before, &Then, &Else);
        Value *TV = Emit(K, Then);
        Value *EV = Branch(K + 1, Else);
        if (A->getType()->isVoidTy())
          return nullptr;
        PHINode *R = PHINode::Create(A->getType(), 2, "", Before->getIterator());
        R->addIncoming(TV, Then->getParent());
        R->addIncoming(EV, Else->getParent());
        return R;
      };
      if (Value *R = Branch(0, A))
        A->replaceAllUsesWith(R);
      A->eraseFromParent();
    }
    for (Instruction *I : reverse(Work))
      I->eraseFromParent();
    return true;
  }

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    SmallVector<Instruction *, 4> Mixed;
    for (Instruction &I : instructions(F))
      if (isa<PHINode, SelectInst>(I) && I.getType()->isPointerTy() &&
          I.getType()->getPointerAddressSpace() == SPIRV_GENERIC_AS)
        Mixed.push_back(&I);
    bool Changed = false;
    for (Instruction *I : Mixed)
      Changed |= split(I);
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Inlines every function that takes or returns a pointer: a shader has no
// pointers to pass.
class HipVulkanInlinePointerFunctionsPass
    : public HipPassInfoMixin<HipVulkanInlinePointerFunctionsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool Changed = false;
    // Workgroup variables belong to the kernel: what reaches one is inlined.
    SmallSetVector<Function *, 8> Shared;
    std::function<void(Value *)> Walk = [&](Value *V) {
      for (User *U : V->users())
        if (auto *I = dyn_cast<Instruction>(U))
          Shared.insert(I->getFunction());
        else if (isa<ConstantExpr>(U))
          Walk(U);
    };
    for (GlobalVariable &GV : M.globals())
      if (GV.getAddressSpace() == SPIRV_WORKGROUP_AS)
        Walk(&GV);
    for (unsigned I = 0; I < Shared.size(); ++I)
      for (User *U : Shared[I]->users())
        if (auto *CB = dyn_cast<CallBase>(U))
          Shared.insert(CB->getFunction());
    for (Function &F : M) {
      if (F.isDeclaration() || F.getCallingConv() == CallingConv::SPIR_KERNEL)
        continue;
      bool HasPtr = Shared.contains(&F) || F.getReturnType()->isPointerTy() ||
                    any_of(F.args(), [](Argument &A) {
                      return A.getType()->isPointerTy();
                    });
      if (!HasPtr)
        continue;
      F.removeFnAttr(Attribute::NoInline);
      F.removeFnAttr(Attribute::OptimizeNone);
      F.addFnAttr(Attribute::AlwaysInline);
      Changed = true;
      // libclc's Vulkan build takes plain pointers where HIP declares generic
      // ones; such calls are only inlined once their types match.
      for (User *U : make_early_inc_range(F.users())) {
        auto *CI = dyn_cast<CallInst>(U);
        if (!CI || CI->getCalledOperand() != &F ||
            CI->getFunctionType() == F.getFunctionType() || F.isVarArg() ||
            CI->arg_size() != F.arg_size() ||
            CI->getType() != F.getReturnType() ||
            any_of(zip(CI->args(), F.args()), [](auto VA) {
              Type *T = std::get<0>(VA)->getType(), *P = std::get<1>(VA).getType();
              return T != P && !(T->isPointerTy() && P->isPointerTy());
            }))
          continue;
        SmallVector<Value *, 4> Args;
        for (auto [V, A] : zip(CI->args(), F.args()))
          Args.push_back(V->getType() == A.getType()
                             ? V.get()
                             : new AddrSpaceCastInst(V, A.getType(), "",
                                                     CI->getIterator()));
        auto *New = CallInst::Create(&F, Args, "", CI->getIterator());
        New->setCallingConv(CI->getCallingConv());
        New->takeName(CI);
        CI->replaceAllUsesWith(New);
        CI->eraseFromParent();
      }
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Lowers the OpenCL builtins clspv provides natively, which libclc's Vulkan
// build therefore leaves undefined, to LLVM IR the SPIR-V backend handles.
class HipVulkanLowerOpenCLBuiltinsPass
    : public HipPassInfoMixin<HipVulkanLowerOpenCLBuiltinsPass> {
  // Splits an Itanium-mangled C function name into its name and the first
  // parameter's type code.
  static bool demangle(StringRef Mangled, StringRef &Name, char &Param) {
    if (!Mangled.consume_front("_Z")) {
      Name = Mangled;
      Param = 0;
      return true;
    }
    unsigned Len;
    if (Mangled.consumeInteger(10, Len) || Len > Mangled.size())
      return false;
    Name = Mangled.take_front(Len);
    StringRef Rest = Mangled.drop_front(Len);
    while (Rest.consume_front("Dv")) // vector: Dv<N>_<elem>
      Rest = Rest.drop_until([](char C) { return C == '_'; }).drop_front();
    while (Rest.consume_front("PU3AS") || Rest.consume_front("P") ||
           Rest.consume_front("V") || Rest.consume_front("K"))
      Rest = Rest.drop_while([](char C) { return isDigit(C); });
    Rest.consume_front("U7_Atomic");
    Param = Rest.empty() ? 0 : Rest.front();
    return true;
  }
  static bool isSignedCode(char C) {
    return C == 'i' || C == 'l' || C == 'c' || C == 's' || C == 'a' || C == 'x';
  }

public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    LLVMContext &C = M.getContext();
    SyncScope::ID Device = C.getOrInsertSyncScopeID("device");
    bool Changed = false;
    for (Function &F : make_early_inc_range(M.functions())) {
      StringRef Name;
      char Param;
      if (!demangle(F.getName(), Name, Param))
        continue;
      // chipStar's warp ballot, defined through sub-group builtins, and
      // libclc's software fma, whose control flow the backend cannot
      // structure.
      if ((Name == "__chip_ballot" || Name == "__chip_all" ||
           Name == "__chip_any" || Name == "__chip_syncwarp" || Name == "fma" ||
           Name == "__clc_fma") &&
          !F.isDeclaration())
        F.deleteBody();
      if (!F.isDeclaration())
        continue;
      bool Signed = isSignedCode(Param);
      auto Lower = [&](function_ref<Value *(IRBuilder<> &, CallInst *)> Emit) {
        for (User *U : make_early_inc_range(F.users()))
          if (auto *CI = dyn_cast<CallInst>(U);
              CI && CI->getCalledFunction() == &F) {
            IRBuilder<> B(CI);
            if (Value *V = Emit(B, CI))
              CI->replaceAllUsesWith(V);
            CI->eraseFromParent();
            Changed = true;
          }
      };
      auto Arg = [](CallInst *CI, unsigned I) { return CI->getArgOperand(I); };
      static const StringMap<Intrinsic::ID> Unary = {
          {"floor", Intrinsic::floor}, {"ceil", Intrinsic::ceil},
          {"trunc", Intrinsic::trunc}, {"rint", Intrinsic::rint},
          {"round", Intrinsic::round}, {"fabs", Intrinsic::fabs},
          {"sqrt", Intrinsic::sqrt},   {"popcount", Intrinsic::ctpop}};
      static const StringMap<Intrinsic::ID> Binary = {
          {"fmin", Intrinsic::minnum},
          {"fmax", Intrinsic::maxnum},
          {"copysign", Intrinsic::copysign}};
      if (auto It = Unary.find(Name); It != Unary.end()) {
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          return B.CreateUnaryIntrinsic(It->second, Arg(CI, 0));
        });
      } else if (auto It2 = Binary.find(Name); It2 != Binary.end()) {
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          return B.CreateBinaryIntrinsic(It2->second, Arg(CI, 0), Arg(CI, 1));
        });
      } else if (Name == "fma" || Name == "__clc_fma" || Name == "mad") {
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          return B.CreateIntrinsic(CI->getType(),
                                   Name == "mad" ? Intrinsic::fmuladd
                                                 : Intrinsic::fma,
                                   {Arg(CI, 0), Arg(CI, 1), Arg(CI, 2)});
        });
      } else if (Name == "llvm.get.rounding") {
        // Round to nearest, the only mode HIP code runs in.
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          return ConstantInt::get(CI->getType(), 1);
        });
      } else if (Name == "llvm.readcyclecounter" ||
                 Name == "llvm.readsteadycounter") {
        // Vulkan's shader clock, at the scope devices support most. Drivers
        // may delete a clock read nothing uses (Mesa's NIR marks it
        // CAN_ELIMINATE), and with it a busy-wait loop; an atomic keeps it.
        FunctionCallee Clock = M.getOrInsertFunction(
            "_Z20__spirv_ReadClockKHRi",
            FunctionType::get(Type::getInt64Ty(C), {Type::getInt32Ty(C)},
                              false));
        Type *I32 = Type::getInt32Ty(C);
        auto *Keep = cast<GlobalVariable>(M.getOrInsertGlobal(
            "__chip_clock_keep", I32, [&] {
              return new GlobalVariable(M, I32, false,
                                        GlobalValue::InternalLinkage,
                                        PoisonValue::get(I32),
                                        "__chip_clock_keep", nullptr,
                                        GlobalValue::NotThreadLocal,
                                        SPIRV_WORKGROUP_AS);
            }));
        Lower([&](IRBuilder<> &B, CallInst *) {
          Value *T = B.CreateCall(Clock, {B.getInt32(/*Subgroup=*/3)});
          B.CreateAtomicRMW(AtomicRMWInst::Add, Keep, B.getInt32(1), Align(4),
                            AtomicOrdering::Monotonic,
                            C.getOrInsertSyncScopeID("workgroup"));
          return T;
        });
      } else if (Name == "hadd" || Name == "rhadd") {
        // (x >> 1) + (y >> 1) + the carry of the low bits, without overflow.
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          Value *X = Arg(CI, 0), *Y = Arg(CI, 1);
          auto Half = [&](Value *V) {
            return Signed ? B.CreateAShr(V, 1) : B.CreateLShr(V, 1);
          };
          Value *Low = Name == "hadd" ? B.CreateAnd(X, Y) : B.CreateOr(X, Y);
          return B.CreateAdd(
              B.CreateAdd(Half(X), Half(Y)),
              B.CreateAnd(Low, ConstantInt::get(X->getType(), 1)));
        });
      } else if (Name == "mul24") {
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          auto Low24 = [&](Value *V) {
            return Signed ? B.CreateAShr(B.CreateShl(V, 8), 8)
                          : B.CreateAnd(V, 0xffffff);
          };
          return B.CreateMul(Low24(Arg(CI, 0)), Low24(Arg(CI, 1)));
        });
      } else if (Name == "mul_hi" &&
                 F.getReturnType()->getScalarSizeInBits() == 64) {
        // From 32-bit halves: Vulkan has no 128-bit integers.
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          Value *X = Arg(CI, 0), *Y = Arg(CI, 1);
          Type *T = X->getType();
          Constant *Lo = ConstantInt::get(T, 0xffffffffu), *S32 = ConstantInt::get(T, 32);
          Value *XL = B.CreateAnd(X, Lo), *XH = B.CreateLShr(X, S32);
          Value *YL = B.CreateAnd(Y, Lo), *YH = B.CreateLShr(Y, S32);
          Value *LH = B.CreateMul(XL, YH), *HL = B.CreateMul(XH, YL);
          Value *Mid = B.CreateAdd(
              B.CreateAdd(B.CreateLShr(B.CreateMul(XL, YL), S32),
                          B.CreateAnd(LH, Lo)),
              B.CreateAnd(HL, Lo));
          Value *Hi = B.CreateAdd(
              B.CreateAdd(B.CreateMul(XH, YH), B.CreateLShr(LH, S32)),
              B.CreateAdd(B.CreateLShr(HL, S32), B.CreateLShr(Mid, S32)));
          if (!Signed)
            return Hi;
          Value *Zero = Constant::getNullValue(T);
          auto IfNeg = [&](Value *V, Value *W) {
            return B.CreateSelect(B.CreateICmpSLT(V, Zero), W, Zero);
          };
          return B.CreateSub(B.CreateSub(Hi, IfNeg(X, Y)), IfNeg(Y, X));
        });
      } else if (Name == "mul_hi") {
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          Type *T = CI->getType();
          Type *Wide = T->getWithNewBitWidth(2 * T->getScalarSizeInBits());
          auto Ext = [&](Value *V) {
            return Signed ? B.CreateSExt(V, Wide) : B.CreateZExt(V, Wide);
          };
          return B.CreateTrunc(
              B.CreateLShr(B.CreateMul(Ext(Arg(CI, 0)), Ext(Arg(CI, 1))),
                           T->getScalarSizeInBits()),
              T);
        });
      } else if (Name.starts_with("native_") &&
                 is_contained({"sin", "cos", "exp", "exp2", "log", "log2",
                               "sqrt", "rsqrt", "recip", "divide", "tan",
                               "exp10", "log10", "powr"},
                              Name.drop_front(strlen("native_")))) {
        // Fast, imprecise variants: the GLSL instructions the backend emits.
        StringRef Op = Name.drop_front(strlen("native_"));
        static const StringMap<Intrinsic::ID> Direct = {
            {"sin", Intrinsic::sin},   {"cos", Intrinsic::cos},
            {"exp", Intrinsic::exp},   {"exp2", Intrinsic::exp2},
            {"log", Intrinsic::log},   {"log2", Intrinsic::log2},
            {"sqrt", Intrinsic::sqrt}};
        Lower([&](IRBuilder<> &B, CallInst *CI) -> Value * {
          Value *X = Arg(CI, 0);
          Type *T = X->getType();
          auto U = [&](Intrinsic::ID ID, Value *V) {
            return B.CreateUnaryIntrinsic(ID, V);
          };
          auto C = [&](double V) { return ConstantFP::get(T, V); };
          if (auto It = Direct.find(Op); It != Direct.end())
            return U(It->second, X);
          if (Op == "rsqrt")
            return B.CreateFDiv(C(1), U(Intrinsic::sqrt, X));
          if (Op == "recip")
            return B.CreateFDiv(C(1), X);
          if (Op == "divide")
            return B.CreateFDiv(X, Arg(CI, 1));
          if (Op == "tan")
            return B.CreateFDiv(U(Intrinsic::sin, X), U(Intrinsic::cos, X));
          if (Op == "exp10")
            return U(Intrinsic::exp2, B.CreateFMul(X, C(numbers::ln10 / numbers::ln2)));
          if (Op == "log10")
            return B.CreateFMul(U(Intrinsic::log2, X), C(numbers::ln2 / numbers::ln10));
          assert(Op == "powr");
          return U(Intrinsic::exp2,
                   B.CreateFMul(Arg(CI, 1), U(Intrinsic::log2, X)));
        });
      } else if (Name == "convert_float" || Name == "convert_double") {
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          Value *X = Arg(CI, 0);
          if (X->getType()->isFPOrFPVectorTy())
            return B.CreateFPCast(X, CI->getType());
          return Signed ? B.CreateSIToFP(X, CI->getType())
                        : B.CreateUIToFP(X, CI->getType());
        });
      } else if (Name == "isnan" || Name == "isinf" || Name == "isfinite") {
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          Value *X = Arg(CI, 0);
          Value *Inf = ConstantFP::getInfinity(X->getType());
          Value *R = Name == "isnan"
                         ? B.CreateFCmpUNO(X, X)
                         : Name == "isinf" ? B.CreateFCmpOEQ(B.CreateUnaryIntrinsic(
                                                                 Intrinsic::fabs, X),
                                                             Inf)
                                           : B.CreateFCmpOLT(B.CreateUnaryIntrinsic(
                                                                 Intrinsic::fabs, X),
                                                             Inf);
          // A scalar is true as 1, a vector lane as -1.
          return CI->getType()->isVectorTy() ? B.CreateSExt(R, CI->getType())
                                             : B.CreateZExt(R, CI->getType());
        });
      } else if (Name == "clz" || Name == "ctz") {
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          return B.CreateBinaryIntrinsic(
              Name == "clz" ? Intrinsic::ctlz : Intrinsic::cttz, Arg(CI, 0),
              B.getFalse());
        });
      } else if (Name == "min" || Name == "max") {
        bool FP = F.getReturnType()->isFPOrFPVectorTy();
        Intrinsic::ID ID =
            FP ? (Name == "min" ? Intrinsic::minnum : Intrinsic::maxnum)
               : Name == "min" ? (Signed ? Intrinsic::smin : Intrinsic::umin)
                               : (Signed ? Intrinsic::smax : Intrinsic::umax);
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          return B.CreateBinaryIntrinsic(ID, Arg(CI, 0), Arg(CI, 1));
        });
      } else if (Name == "clamp") {
        bool FP = F.getReturnType()->isFPOrFPVectorTy();
        Intrinsic::ID Max = FP ? Intrinsic::maxnum
                               : Signed ? Intrinsic::smax : Intrinsic::umax;
        Intrinsic::ID Min = FP ? Intrinsic::minnum
                               : Signed ? Intrinsic::smin : Intrinsic::umin;
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          return B.CreateBinaryIntrinsic(
              Min, B.CreateBinaryIntrinsic(Max, Arg(CI, 0), Arg(CI, 1)),
              Arg(CI, 2));
        });
      } else if (Name == "barrier" || Name == "work_group_barrier") {
        Lower([&](IRBuilder<> &B, CallInst *) -> Value * {
          B.CreateIntrinsic(Intrinsic::spv_all_memory_barrier_with_group_sync,
                            {});
          return nullptr;
        });
      } else if (Name == "atomic_work_item_fence" || Name == "mem_fence") {
        Lower([&](IRBuilder<> &B, CallInst *) -> Value * {
          B.CreateFence(AtomicOrdering::SequentiallyConsistent, Device);
          return nullptr;
        });
      } else if (Name == "get_sub_group_local_id" ||
                 Name == "get_sub_group_size" ||
                 Name == "get_sub_group_id" ||
                 Name == "get_max_sub_group_size") {
        Intrinsic::ID ID =
            Name == "get_sub_group_local_id"
                ? Intrinsic::spv_subgroup_local_invocation_id
            : Name == "get_sub_group_id" ? Intrinsic::spv_subgroup_id
            : Name == "get_sub_group_size" ? Intrinsic::spv_subgroup_size
                                             : Intrinsic::spv_subgroup_max_size;
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          return B.CreateZExtOrTrunc(B.CreateIntrinsic(B.getInt32Ty(), ID, {}),
                                     CI->getType());
        });
      } else if (Name == "get_global_size") {
        // The SPIR-V backend lowers the work-item builtins it is built from.
        FunctionType *FT = F.getFunctionType();
        FunctionCallee Groups = M.getOrInsertFunction("_Z14get_num_groupsj", FT);
        FunctionCallee Size = M.getOrInsertFunction("_Z14get_local_sizej", FT);
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          return B.CreateMul(B.CreateCall(Groups, Arg(CI, 0)),
                             B.CreateCall(Size, Arg(CI, 0)));
        });
      } else if (Name == "sub_group_shuffle") {
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          return B.CreateIntrinsic(
              CI->getType(), Intrinsic::spv_wave_readlane,
              {Arg(CI, 0), B.CreateZExtOrTrunc(Arg(CI, 1), B.getInt32Ty())});
        });
      } else if (Name == "work_group_all" || Name == "work_group_any" ||
                 Name == "work_group_reduce_add") {
        Lower([&](IRBuilder<> &B, CallInst *CI) -> Value * {
          Value *X = Arg(CI, 0);
          if (Name == "work_group_reduce_add")
            return B.CreateSExtOrTrunc(
                blockSum(M, B, B.CreateSExtOrTrunc(X, B.getInt32Ty())),
                CI->getType());
          Value *P = B.CreateICmpNE(X, Constant::getNullValue(X->getType()));
          if (Name == "work_group_all")
            P = B.CreateNot(P);
          Value *N = blockSum(M, B, B.CreateZExt(P, B.getInt32Ty()));
          Value *R = Name == "work_group_all"
                         ? B.CreateICmpEQ(N, B.getInt32(0))
                         : B.CreateICmpNE(N, B.getInt32(0));
          return B.CreateZExt(R, CI->getType());
        });
      } else if (Name == "__chip_ballot") {
        // Warp ballot: the low 64 bits of the sub-group mask.
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          Value *Pred = B.CreateICmpNE(
              Arg(CI, 0), Constant::getNullValue(Arg(CI, 0)->getType()));
          Value *Mask =
              B.CreateIntrinsic(Intrinsic::spv_subgroup_ballot, {Pred});
          Value *Lo = B.CreateZExt(B.CreateExtractElement(Mask, uint64_t(0)),
                                   B.getInt64Ty());
          Value *Hi = B.CreateZExt(B.CreateExtractElement(Mask, uint64_t(1)),
                                   B.getInt64Ty());
          return B.CreateZExtOrTrunc(B.CreateOr(Lo, B.CreateShl(Hi, 32)),
                                     CI->getType());
        });
      } else if (Name == "__chip_all" || Name == "__chip_any") {
        // Over the lanes the sub-group really has, which a ballot compared
        // with a 32-lane mask is not.
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          Value *Pred = B.CreateICmpNE(
              Arg(CI, 0), Constant::getNullValue(Arg(CI, 0)->getType()));
          return B.CreateZExt(
              B.CreateIntrinsic(Name == "__chip_all" ? Intrinsic::spv_wave_all
                                                     : Intrinsic::spv_wave_any,
                                {Pred}),
              CI->getType());
        });
      } else if (Name == "__chip_syncwarp") {
        // Lanes of a sub-group execute together; what callers need is that
        // their memory accesses are ordered.
        Lower([&](IRBuilder<> &B, CallInst *) -> Value * {
          B.CreateIntrinsic(Intrinsic::spv_all_memory_barrier, {});
          return nullptr;
        });
      } else if (Name == "to_local" || Name == "__to_local" ||
                 Name == "to_global" || Name == "__to_global") {
        unsigned AS = Name.contains("local") ? 3 : 1;
        Lower([&](IRBuilder<> &B, CallInst *CI) {
          return B.CreateAddrSpaceCast(Arg(CI, 0), B.getPtrTy(AS));
        });
      } else if ((Name.consume_front("atomic_") ||
                  Name.consume_front("atom_")) &&
                 (Name.consume_back("_explicit") || true)) {
        // OpenCL 1.x names.
        static const StringMap<StringRef> Legacy = {
            {"add", "fetch_add"}, {"sub", "fetch_sub"}, {"and", "fetch_and"},
            {"or", "fetch_or"},   {"xor", "fetch_xor"}, {"min", "fetch_min"},
            {"max", "fetch_max"}, {"xchg", "exchange"}};
        if (auto It = Legacy.find(Name); It != Legacy.end())
          Name = It->second;
        bool FP = false;
        Type *ValTy = F.getReturnType();
        std::optional<AtomicRMWInst::BinOp> Op;
        if (Name == "fetch_add" || Name == "fetch_sub") {
          FP = ValTy->isFloatingPointTy();
          Op = Name == "fetch_add"
                   ? (FP ? AtomicRMWInst::FAdd : AtomicRMWInst::Add)
                   : (FP ? AtomicRMWInst::FSub : AtomicRMWInst::Sub);
        } else if (Name == "fetch_and")
          Op = AtomicRMWInst::And;
        else if (Name == "fetch_or")
          Op = AtomicRMWInst::Or;
        else if (Name == "fetch_xor")
          Op = AtomicRMWInst::Xor;
        else if (Name == "fetch_min" || Name == "fetch_max") {
          FP = ValTy->isFloatingPointTy();
          Op = Name == "fetch_min"
                   ? (FP ? AtomicRMWInst::FMin
                         : Signed ? AtomicRMWInst::Min : AtomicRMWInst::UMin)
                   : (FP ? AtomicRMWInst::FMax
                         : Signed ? AtomicRMWInst::Max : AtomicRMWInst::UMax);
        } else if (Name == "exchange")
          Op = AtomicRMWInst::Xchg;
        if (Op) {
          Lower([&](IRBuilder<> &B, CallInst *CI) {
            return B.CreateAtomicRMW(*Op, Arg(CI, 0), Arg(CI, 1), MaybeAlign(),
                                     AtomicOrdering::Monotonic, Device);
          });
        } else if (Name == "load") {
          Lower([&](IRBuilder<> &B, CallInst *CI) {
            LoadInst *L = B.CreateLoad(CI->getType(), Arg(CI, 0));
            L->setAtomic(AtomicOrdering::Monotonic, Device);
            L->setAlignment(Align(CI->getType()->getPrimitiveSizeInBits() / 8));
            return L;
          });
        } else if (Name == "store") {
          Lower([&](IRBuilder<> &B, CallInst *CI) -> Value * {
            StoreInst *S = B.CreateStore(Arg(CI, 1), Arg(CI, 0));
            S->setAtomic(AtomicOrdering::Monotonic, Device);
            S->setAlignment(
                Align(Arg(CI, 1)->getType()->getPrimitiveSizeInBits() / 8));
            return nullptr;
          });
        } else if (Name == "inc" || Name == "dec") {
          Lower([&](IRBuilder<> &B, CallInst *CI) {
            return B.CreateAtomicRMW(
                Name == "inc" ? AtomicRMWInst::Add : AtomicRMWInst::Sub,
                Arg(CI, 0), ConstantInt::get(CI->getType(), 1), MaybeAlign(),
                AtomicOrdering::Monotonic, Device);
          });
        } else if (Name == "cmpxchg") {
          // (object, expected, desired): the value found.
          Lower([&](IRBuilder<> &B, CallInst *CI) {
            return B.CreateExtractValue(
                B.CreateAtomicCmpXchg(Arg(CI, 0), Arg(CI, 1), Arg(CI, 2),
                                      MaybeAlign(), AtomicOrdering::Monotonic,
                                      AtomicOrdering::Monotonic, Device),
                0);
          });
        } else if (Name.starts_with("compare_exchange")) {
          // (object, expected*, desired, ...): true if *expected matched,
          // else *expected = the value found.
          Lower([&](IRBuilder<> &B, CallInst *CI) {
            Type *T = Arg(CI, 2)->getType();
            Value *Expected = B.CreateLoad(T, Arg(CI, 1));
            Value *Pair = B.CreateAtomicCmpXchg(
                Arg(CI, 0), Expected, Arg(CI, 2), MaybeAlign(),
                AtomicOrdering::Monotonic, AtomicOrdering::Monotonic, Device);
            B.CreateStore(B.CreateExtractValue(Pair, 0), Arg(CI, 1));
            return B.CreateZExt(B.CreateExtractValue(Pair, 1), CI->getType());
          });
        }
      }
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// For the shadervulkan environment the SPIR-V backend lowers the kernels
// itself; it needs helpers inlined, generic pointers resolved and nothing but
// the kernels left with external linkage.
static void addVulkanLinkTimePasses(ModulePassManager &MPM) {
#ifndef CHIP_KEEP_KERNEL_DEBUG_INFO
  MPM.addPass(HipStripDebugInfoPass());
#endif
  MPM.addPass(RemoveNoInlineOptNoneAttrsPass());
  MPM.addPass(HipEmitLoweredNamesPass());
  MPM.addPass(HipStripUsedIntrinsicsPass());
  MPM.addPass(HipAbortPass());
  MPM.addPass(HipVulkanLowerBuiltinsPass());
  MPM.addPass(HipVulkanLowerOpenCLBuiltinsPass());
  MPM.addPass(HipSharedAddrLocalInitPass());
  MPM.addPass(HipVulkanDefineDynamicSharedPass());
  // Internal helpers, weak ones included, can be inlined and then dropped, so
  // that kernels are the only users of the device globals that become their
  // arguments.
  MPM.addPass(InternalizePass([](const GlobalValue &GV) {
    if (const auto *F = dyn_cast<Function>(&GV))
      return F->getCallingConv() == CallingConv::SPIR_KERNEL;
    const auto *V = dyn_cast<GlobalVariable>(&GV);
    return V && V->isExternallyInitialized();
  }));
  MPM.addPass(HipVulkanInlinePointerFunctionsPass());
  MPM.addPass(AlwaysInlinerPass());
  MPM.addPass(ModuleInlinerWrapperPass(getInlineParams(1000)));
  // printf needs its constant strings out of the -O0 argument spills.
  MPM.addPass(createModuleToFunctionPassAdaptor(SROAPass(SROAOptions::ModifyCFG)));
  MPM.addPass(HipVulkanLowerPrintfPass());
  MPM.addPass(GlobalDCEPass());
  MPM.addPass(HipVulkanExposeStaticGlobalsPass());
  MPM.addPass(HipVulkanFoldPointerGlobalsPass());
  MPM.addPass(HipVulkanDropUnusedAbortFlagPass());
  MPM.addPass(HipGlobalVariablesPass());
  MPM.addPass(HipVulkanLowerOpenCLBuiltinsPass()); // Used by its shadow kernels.
  MPM.addPass(HipVulkanSplitPointerFieldsPass());
  MPM.addPass(HipVulkanPrivatizeConstantsPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(SROAPass(SROAOptions::ModifyCFG)));
  MPM.addPass(createModuleToFunctionPassAdaptor(HipVulkanFoldPtrIntPairsPass()));
  MPM.addPass(createModuleToFunctionPassAdaptor(HipVulkanPointerTablesPass()));
  MPM.addPass(createModuleToFunctionPassAdaptor(InferAddressSpacesPass(4u)));
  MPM.addPass(createModuleToFunctionPassAdaptor(HipVulkanResolveAddrSpaceCastsPass()));
  MPM.addPass(createModuleToFunctionPassAdaptor(HipVulkanExpandMemTransfersPass()));
  MPM.addPass(createModuleToFunctionPassAdaptor(HipLowerMemsetPass()));
  MPM.addPass(createModuleToFunctionPassAdaptor(HipVulkanSplitBufferChoicesPass()));
  MPM.addPass(createModuleToFunctionPassAdaptor(HipVulkanSplitMixedPointersPass()));
  MPM.addPass(HipVulkanRetypeArraysPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(SROAPass(SROAOptions::ModifyCFG)));
  // SROA leaves the integer bits of split pointer fields dead.
  MPM.addPass(createModuleToFunctionPassAdaptor(DCEPass()));
  MPM.addPass(HipVulkanLockSharedFloatAtomicsPass());
  // Devices need not support floating-point atomic min/max (the B570 has none
  // for double).
  MPM.addPass(createModuleToFunctionPassAdaptor(HipLowerFPAtomicMinMaxPass()));
  MPM.addPass(createModuleToFunctionPassAdaptor(HipVulkanPreciseDivPass()));
  // A Volatile memory operand does not make another invocation's store visible
  // under the GLSL450 memory model, so volatile becomes a relaxed atomic.
  MPM.addPass(createModuleToFunctionPassAdaptor(HipLowerVolatileAccessesPass()));
  MPM.addPass(InternalizePass([](const GlobalValue &GV) {
    const auto *F = dyn_cast<Function>(&GV);
    return F && F->getCallingConv() == CallingConv::SPIR_KERNEL;
  }));
  MPM.addPass(GlobalDCEPass());
  MPM.addPass(HipDropUnusedGlobalsPass());
  MPM.addPass(HipVulkanDropDeadHiddenArgsPass());
  // WORKAROUND(CHIP-SPV/chipStar#1738, no upstream report): the SPIR-V
  // structurizer gives a branch and a switch sharing a target a merge block
  // their selection header does not dominate. Remove when
  // CanaryVulkanStructurizerMergeDominance fires.
  MPM.addPass(createModuleToFunctionPassAdaptor(LowerSwitchPass()));
  MPM.addPass(createModuleToFunctionPassAdaptor(StructurizeCFGPass()));
}
#endif

// Runs the Vulkan pipeline for the shadervulkan environment and the OpenCL one
// otherwise; the module is not known when the pipeline is built.
class HipPostLinkPipelinePass
    : public HipPassInfoMixin<HipPostLinkPipelinePass> {
  ModulePassManager OpenCL, Vulkan;

public:
  HipPostLinkPipelinePass() {
    addFullLinkTimePasses(OpenCL);
#if LLVM_VERSION_MAJOR >= 24
    addVulkanLinkTimePasses(Vulkan);
#endif
  }
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
#if LLVM_VERSION_MAJOR >= 24
    if (Triple(M.getTargetTriple()).getEnvironmentName() == "shadervulkan")
      return Vulkan.run(M, AM);
#endif
    return OpenCL.run(M, AM);
  }
};

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
                    MPM.addPass(HipPostLinkPipelinePass());
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
