//===- HipLowerSubwordAtomics.cpp -----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Clang lowers __hip_atomic_load / store / exchange / fetch_* /
// compare_exchange on char and short (and the atomic builtins on half and
// bfloat16) to `load atomic i8`, `store atomic i16`, `atomicrmw ... i8` and
// `cmpxchg i16`. Both SPIR-V producers pass those through as OpAtomicLoad,
// OpAtomicStore, OpAtomicIAdd, OpAtomicCompareExchange ... on OpTypeInt 8 / 16,
// and OpenCL SPIR-V consumers only implement 32 and 64 bit atomics: IGC fails
// the module build with
//
//   error: undefined reference to `_Z18__spirv_AtomicLoadPU3AS4cii'
//   error: backend compiler failed build.
//
// and the Intel CPU runtime with "JIT session error: Symbols not found:
// [ _Z20atomic_load_explicitPU3AS1VU7_Atomicc12memory_order12memory_scope ...".
// Either failure takes every kernel in the module down with it.
//
// Do what LLVM's AtomicExpand does for targets without narrow atomics: operate
// on the aligned 32 bit word that contains the value and touch only its lane.
//
//   load     -> 32 bit atomic load, shift, truncate
//   store    -> cmpxchg loop that replaces the lane and keeps the other bytes
//   atomicrmw-> cmpxchg loop that applies the operation to the lane only
//   cmpxchg  -> 32 bit cmpxchg with the expected and new values masked into
//               the current word, retried while only the other lanes changed
//
// Every generated atomic keeps the original ordering and syncscope; the loops'
// initial read is a monotonic atomic load in the same scope. Values are
// assumed to be naturally aligned (which is what clang emits): a 16 bit value
// at an odd address could straddle two words and is left untouched.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerSubwordAtomics.h"

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include "PassPluginCompat.h"

#define PASS_NAME "hip-lower-subword-atomics"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

namespace {

constexpr unsigned WordBytes = 4;

/// Where the lane of one 8 or 16 bit value sits inside its containing word.
struct LaneInWord {
  Value *WordAddr = nullptr; // The aligned word, in the original address space.
  Value *Shift = nullptr;    // i32: bit offset of the lane inside the word.
  Value *Mask = nullptr;     // i32: ones over the lane.
  Value *InvMask = nullptr;  // i32: ones over the other lanes.
  Type *ValueTy = nullptr;   // i8, i16, half or bfloat.
  Type *IntTy = nullptr;     // i8 or i16.
};

bool isSubwordType(Type *Ty) {
  if (!Ty->isIntegerTy() && !Ty->isFloatingPointTy())
    return false;
  unsigned Bits = Ty->getPrimitiveSizeInBits();
  return Bits == 8 || Bits == 16;
}

/// Emit the address of the word containing `Ptr` and the lane's position in
/// it. The word address is formed with a GEP rather than ptrtoint / inttoptr so
/// the pointer keeps its address space and provenance.
LaneInWord locateLane(IRBuilder<> &B, Value *Ptr, Align PtrAlign,
                      Type *ValueTy, const DataLayout &DL) {
  LLVMContext &Ctx = B.getContext();
  Type *Int32Ty = B.getInt32Ty();
  LaneInWord L;
  L.ValueTy = ValueTy;
  unsigned ValueBits = ValueTy->getPrimitiveSizeInBits();
  L.IntTy = Type::getIntNTy(Ctx, ValueBits);

  if (PtrAlign >= Align(WordBytes)) {
    L.WordAddr = Ptr;
    L.Shift = ConstantInt::get(Int32Ty, 0);
  } else {
    Type *IdxTy = DL.getIndexType(Ptr->getType());
    Value *AddrInt = B.CreatePtrToInt(Ptr, IdxTy);
    Value *ByteInWord = B.CreateAnd(AddrInt, WordBytes - 1, "lane.byte");
    L.WordAddr = B.CreateGEP(B.getInt8Ty(), Ptr, B.CreateNeg(ByteInWord),
                             "word.addr");
    if (!DL.isLittleEndian())
      ByteInWord = B.CreateXor(ByteInWord, WordBytes - ValueBits / 8);
    L.Shift = B.CreateTrunc(B.CreateShl(ByteInWord, 3), Int32Ty, "lane.shift");
  }
  L.Mask = B.CreateShl(ConstantInt::get(Int32Ty, (1u << ValueBits) - 1),
                       L.Shift, "lane.mask");
  L.InvMask = B.CreateNot(L.Mask, "lane.invmask");
  return L;
}

/// The lane's value, as the original type, out of a whole word.
Value *extractLane(IRBuilder<> &B, Value *Word, const LaneInWord &L) {
  Value *Narrow = B.CreateTrunc(B.CreateLShr(Word, L.Shift), L.IntTy);
  return B.CreateBitCast(Narrow, L.ValueTy);
}

/// `V` moved into the lane's position, with every other bit zero.
Value *laneBits(IRBuilder<> &B, Value *V, const LaneInWord &L) {
  Value *Int = B.CreateBitCast(V, L.IntTy);
  return B.CreateShl(B.CreateZExt(Int, B.getInt32Ty()), L.Shift);
}

/// `Word` with its lane replaced by `V`.
Value *insertLane(IRBuilder<> &B, Value *Word, Value *V, const LaneInWord &L) {
  return B.CreateOr(B.CreateAnd(Word, L.InvMask), laneBits(B, V, L));
}

/// The value an atomicrmw leaves in memory, computed on the narrow type.
Value *applyRMWOperation(IRBuilder<> &B, AtomicRMWInst::BinOp Op, Value *Old,
                         Value *Val) {
  switch (Op) {
  case AtomicRMWInst::Add:
    return B.CreateAdd(Old, Val);
  case AtomicRMWInst::Sub:
    return B.CreateSub(Old, Val);
  case AtomicRMWInst::And:
    return B.CreateAnd(Old, Val);
  case AtomicRMWInst::Nand:
    return B.CreateNot(B.CreateAnd(Old, Val));
  case AtomicRMWInst::Or:
    return B.CreateOr(Old, Val);
  case AtomicRMWInst::Xor:
    return B.CreateXor(Old, Val);
  case AtomicRMWInst::Max:
    return B.CreateSelect(B.CreateICmpSGT(Old, Val), Old, Val);
  case AtomicRMWInst::Min:
    return B.CreateSelect(B.CreateICmpSLE(Old, Val), Old, Val);
  case AtomicRMWInst::UMax:
    return B.CreateSelect(B.CreateICmpUGT(Old, Val), Old, Val);
  case AtomicRMWInst::UMin:
    return B.CreateSelect(B.CreateICmpULE(Old, Val), Old, Val);
  case AtomicRMWInst::FAdd:
    return B.CreateFAdd(Old, Val);
  case AtomicRMWInst::FSub:
    return B.CreateFSub(Old, Val);
  case AtomicRMWInst::FMax:
    return B.CreateBinaryIntrinsic(Intrinsic::maxnum, Old, Val);
  case AtomicRMWInst::FMin:
    return B.CreateBinaryIntrinsic(Intrinsic::minnum, Old, Val);
#if LLVM_VERSION_MAJOR >= 16
  case AtomicRMWInst::UIncWrap: {
    Value *Inc = B.CreateAdd(Old, ConstantInt::get(Old->getType(), 1));
    return B.CreateSelect(B.CreateICmpUGE(Old, Val),
                          Constant::getNullValue(Old->getType()), Inc);
  }
  case AtomicRMWInst::UDecWrap: {
    Value *Dec = B.CreateSub(Old, ConstantInt::get(Old->getType(), 1));
    Value *Wrap = B.CreateOr(
        B.CreateICmpEQ(Old, Constant::getNullValue(Old->getType())),
        B.CreateICmpUGT(Old, Val));
    return B.CreateSelect(Wrap, Val, Dec);
  }
#endif
#if LLVM_VERSION_MAJOR >= 20
  case AtomicRMWInst::USubCond:
    return B.CreateSelect(B.CreateICmpUGE(Old, Val), B.CreateSub(Old, Val),
                          Old);
  case AtomicRMWInst::USubSat:
    return B.CreateBinaryIntrinsic(Intrinsic::usub_sat, Old, Val);
#endif
#if LLVM_VERSION_MAJOR >= 21
  case AtomicRMWInst::FMaximum:
    return B.CreateBinaryIntrinsic(Intrinsic::maximum, Old, Val);
  case AtomicRMWInst::FMinimum:
    return B.CreateBinaryIntrinsic(Intrinsic::minimum, Old, Val);
#endif
  default:
    report_fatal_error("HipLowerSubwordAtomics: unsupported atomicrmw "
                       "operation on an 8 or 16 bit value");
  }
}

AtomicOrdering cmpxchgSuccessOrdering(AtomicOrdering O) {
  return O == AtomicOrdering::Unordered ? AtomicOrdering::Monotonic : O;
}

/// Turn the block holding `I` into
///
///   entry:  [lane setup already emitted]
///           %word = load atomic i32, ptr %word.addr monotonic
///           br %loop
///   loop:   %loaded = phi [ %word, %entry ], [ %prev, %loop ]
///           %new = Update(%loaded)
///           %pair = cmpxchg ptr %word.addr, i32 %loaded, i32 %new
///           %prev = extractvalue %pair, 0
///           br (extractvalue %pair, 1), %end, %loop
///   end:    I ...
///
/// and return %loaded, the word as it was right before the successful swap.
/// The builder is left in front of `I`.
Value *buildWordCmpXchgLoop(
    Instruction *I, IRBuilder<> &B, const LaneInWord &L,
    AtomicOrdering Ordering, SyncScope::ID SSID, bool IsVolatile,
    function_ref<Value *(IRBuilder<> &, Value *)> Update) {
  LLVMContext &Ctx = B.getContext();
  Type *Int32Ty = B.getInt32Ty();
  BasicBlock *EntryBB = I->getParent();
  BasicBlock *EndBB =
      EntryBB->splitBasicBlock(I->getIterator(), "subword.atomic.end");
  BasicBlock *LoopBB = BasicBlock::Create(Ctx, "subword.atomic.loop",
                                          EntryBB->getParent(), EndBB);

  // splitBasicBlock left an unconditional branch to EndBB; retarget it.
  EntryBB->getTerminator()->eraseFromParent();
  B.SetInsertPoint(EntryBB);
  LoadInst *Init =
      B.CreateAlignedLoad(Int32Ty, L.WordAddr, Align(WordBytes), "word");
  Init->setAtomic(AtomicOrdering::Monotonic, SSID);
  Init->setVolatile(IsVolatile);
  B.CreateBr(LoopBB);

  B.SetInsertPoint(LoopBB);
  PHINode *Loaded = B.CreatePHI(Int32Ty, 2, "word.loaded");
  Loaded->addIncoming(Init, EntryBB);
  Value *New = Update(B, Loaded);
  AtomicOrdering Success = cmpxchgSuccessOrdering(Ordering);
  AtomicCmpXchgInst *Pair = B.CreateAtomicCmpXchg(
      L.WordAddr, Loaded, New, Align(WordBytes), Success,
      AtomicCmpXchgInst::getStrongestFailureOrdering(Success), SSID);
  Pair->setVolatile(IsVolatile);
  Value *Prev = B.CreateExtractValue(Pair, 0, "word.prev");
  Value *Done = B.CreateExtractValue(Pair, 1, "word.done");
  Loaded->addIncoming(Prev, LoopBB);
  B.CreateCondBr(Done, EndBB, LoopBB);

  B.SetInsertPoint(I);
  return Loaded;
}

void lowerLoad(LoadInst *LI, const DataLayout &DL) {
  IRBuilder<> B(LI);
  LaneInWord L = locateLane(B, LI->getPointerOperand(), LI->getAlign(),
                            LI->getType(), DL);
  LoadInst *Word =
      B.CreateAlignedLoad(B.getInt32Ty(), L.WordAddr, Align(WordBytes), "word");
  Word->setAtomic(LI->getOrdering(), LI->getSyncScopeID());
  Word->setVolatile(LI->isVolatile());
  LI->replaceAllUsesWith(extractLane(B, Word, L));
  LI->eraseFromParent();
}

void lowerStore(StoreInst *SI, const DataLayout &DL) {
  IRBuilder<> B(SI);
  Value *Val = SI->getValueOperand();
  LaneInWord L = locateLane(B, SI->getPointerOperand(), SI->getAlign(),
                            Val->getType(), DL);
  Value *ValBits = laneBits(B, Val, L);
  buildWordCmpXchgLoop(SI, B, L, SI->getOrdering(), SI->getSyncScopeID(),
                       SI->isVolatile(), [&](IRBuilder<> &B, Value *Loaded) {
                         return B.CreateOr(B.CreateAnd(Loaded, L.InvMask),
                                           ValBits);
                       });
  SI->eraseFromParent();
}

void lowerRMW(AtomicRMWInst *RMW, const DataLayout &DL) {
  IRBuilder<> B(RMW);
  AtomicRMWInst::BinOp Op = RMW->getOperation();
  Value *Val = RMW->getValOperand();
  LaneInWord L = locateLane(B, RMW->getPointerOperand(), RMW->getAlign(),
                            RMW->getType(), DL);
  Value *ValBits = Op == AtomicRMWInst::Xchg ? laneBits(B, Val, L) : nullptr;
  Value *OldWord = buildWordCmpXchgLoop(
      RMW, B, L, RMW->getOrdering(), RMW->getSyncScopeID(), RMW->isVolatile(),
      [&](IRBuilder<> &B, Value *Loaded) -> Value * {
        if (Op == AtomicRMWInst::Xchg)
          return B.CreateOr(B.CreateAnd(Loaded, L.InvMask), ValBits);
        Value *Old = extractLane(B, Loaded, L);
        return insertLane(B, Loaded, applyRMWOperation(B, Op, Old, Val), L);
      });
  RMW->replaceAllUsesWith(extractLane(B, OldWord, L));
  RMW->eraseFromParent();
}

/// The word-sized cmpxchg cannot tell a mismatch in the lane from a change in
/// the other lanes, so a strong cmpxchg retries while only the other lanes
/// moved:
///
///   entry:   %word = load atomic i32 %word.addr monotonic
///            %init.others = and %word, %invmask
///            br %loop
///   loop:    %others = phi [ %init.others, %entry ], [ %old.others, %failure ]
///            %pair = cmpxchg %word.addr, (%others | %cmp.bits),
///                                        (%others | %new.bits)
///            br (extractvalue %pair, 1), %end, %failure
///   failure: %old.others = and (extractvalue %pair, 0), %invmask
///            br (%others != %old.others), %loop, %end
///   end:     { lane of (extractvalue %pair, 0), extractvalue %pair, 1 }
///
/// A weak cmpxchg may fail spuriously anyway, so it skips the retry.
void lowerCmpXchg(AtomicCmpXchgInst *CI, const DataLayout &DL) {
  IRBuilder<> B(CI);
  LLVMContext &Ctx = B.getContext();
  Type *Int32Ty = B.getInt32Ty();
  Value *Cmp = CI->getCompareOperand();
  LaneInWord L = locateLane(B, CI->getPointerOperand(), CI->getAlign(),
                            Cmp->getType(), DL);
  Value *CmpBits = laneBits(B, Cmp, L);
  Value *NewBits = laneBits(B, CI->getNewValOperand(), L);

  BasicBlock *EntryBB = CI->getParent();
  Function *F = EntryBB->getParent();
  BasicBlock *EndBB =
      EntryBB->splitBasicBlock(CI->getIterator(), "subword.cmpxchg.end");
  BasicBlock *FailureBB =
      CI->isWeak() ? nullptr
                   : BasicBlock::Create(Ctx, "subword.cmpxchg.failure", F,
                                        EndBB);
  BasicBlock *LoopBB = BasicBlock::Create(Ctx, "subword.cmpxchg.loop", F,
                                          FailureBB ? FailureBB : EndBB);

  EntryBB->getTerminator()->eraseFromParent();
  B.SetInsertPoint(EntryBB);
  LoadInst *Init =
      B.CreateAlignedLoad(Int32Ty, L.WordAddr, Align(WordBytes), "word");
  Init->setAtomic(AtomicOrdering::Monotonic, CI->getSyncScopeID());
  Init->setVolatile(CI->isVolatile());
  Value *InitOthers = B.CreateAnd(Init, L.InvMask, "word.others");
  B.CreateBr(LoopBB);

  B.SetInsertPoint(LoopBB);
  PHINode *Others = B.CreatePHI(Int32Ty, 2, "others");
  Others->addIncoming(InitOthers, EntryBB);
  AtomicCmpXchgInst *Pair = B.CreateAtomicCmpXchg(
      L.WordAddr, B.CreateOr(Others, CmpBits), B.CreateOr(Others, NewBits),
      Align(WordBytes), CI->getSuccessOrdering(), CI->getFailureOrdering(),
      CI->getSyncScopeID());
  Pair->setVolatile(CI->isVolatile());
  Pair->setWeak(CI->isWeak());
  Value *OldWord = B.CreateExtractValue(Pair, 0, "word.prev");
  Value *Success = B.CreateExtractValue(Pair, 1, "word.done");
  if (FailureBB) {
    B.CreateCondBr(Success, EndBB, FailureBB);
    B.SetInsertPoint(FailureBB);
    Value *OldOthers = B.CreateAnd(OldWord, L.InvMask, "word.prev.others");
    Value *OnlyOthersMoved = B.CreateICmpNE(Others, OldOthers);
    B.CreateCondBr(OnlyOthersMoved, LoopBB, EndBB);
    Others->addIncoming(OldOthers, FailureBB);
  } else {
    B.CreateBr(EndBB);
  }

  B.SetInsertPoint(CI);
  Value *Res = PoisonValue::get(CI->getType());
  Res = B.CreateInsertValue(Res, extractLane(B, OldWord, L), 0);
  Res = B.CreateInsertValue(Res, Success, 1);
  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();
}

/// A value narrower than its alignment allows can straddle two words, which
/// no single-word operation can serve. Clang never emits such an atomic, so
/// leave it to fail loudly downstream rather than lower it wrongly.
bool isNaturallyAligned(Align A, Type *ValueTy) {
  return A.value() * 8 >= ValueTy->getPrimitiveSizeInBits();
}

bool lowerSubwordAtomics(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  SmallVector<Instruction *, 16> WorkList;
  for (auto &BB : F)
    for (auto &I : BB) {
      Type *Ty = nullptr;
      Align A;
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (!LI->isAtomic())
          continue;
        Ty = LI->getType();
        A = LI->getAlign();
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (!SI->isAtomic())
          continue;
        Ty = SI->getValueOperand()->getType();
        A = SI->getAlign();
      } else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
        Ty = RMW->getType();
        A = RMW->getAlign();
      } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) {
        Ty = CX->getCompareOperand()->getType();
        A = CX->getAlign();
      } else {
        continue;
      }
      if (!isSubwordType(Ty))
        continue;
      if (!isNaturallyAligned(A, Ty)) {
        errs() << "warning: HipLowerSubwordAtomics: leaving under-aligned "
                  "atomic alone in "
               << F.getName() << ": " << I << "\n";
        continue;
      }
      WorkList.push_back(&I);
    }

  for (Instruction *I : WorkList) {
    if (auto *LI = dyn_cast<LoadInst>(I))
      lowerLoad(LI, DL);
    else if (auto *SI = dyn_cast<StoreInst>(I))
      lowerStore(SI, DL);
    else if (auto *RMW = dyn_cast<AtomicRMWInst>(I))
      lowerRMW(RMW, DL);
    else
      lowerCmpXchg(cast<AtomicCmpXchgInst>(I), DL);
  }
  return !WorkList.empty();
}

} // namespace

PreservedAnalyses HipLowerSubwordAtomicsPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  return lowerSubwordAtomics(F) ? PreservedAnalyses::none()
                                : PreservedAnalyses::all();
}

#ifndef CHIP_COMBINED_PASS_PLUGIN
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_NAME, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_NAME) {
                    FPM.addPass(HipLowerSubwordAtomicsPass());
                    return true;
                  }
                  return false;
                });
          }};
}
#endif // CHIP_COMBINED_PASS_PLUGIN
