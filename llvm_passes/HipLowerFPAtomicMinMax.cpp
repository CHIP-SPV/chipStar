//===- HipLowerFPAtomicMinMax.cpp -----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Clang lowers __hip_atomic_fetch_min / __hip_atomic_fetch_max on a floating
// point type to `atomicrmw fmin` / `fmax`. llvm-spirv can only translate those
// with SPV_EXT_shader_atomic_float_min_max, which is not in the extension allow
// list chipStar's driver passes to the translator, so the link fails with
//
//   RequiresExtension: Feature requires the following SPIR-V extension:
//    SPV_EXT_shader_atomic_float_min_max
//
// Widening the allow list would only move the failure to program build time on
// drivers without cl_ext_float_atomics. Expand the operation to a cmpxchg loop
// instead, which every target chipStar supports can run, and which is what
// devicelib.cl already does for atomicMin and atomicMax.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerFPAtomicMinMax.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Passes/PassBuilder.h>
#include "PassPluginCompat.h"

#define PASS_NAME "hip-lower-fp-atomic-min-max"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

static bool isFPMinMax(const AtomicRMWInst &RMW) {
  return RMW.getOperation() == AtomicRMWInst::FMin ||
         RMW.getOperation() == AtomicRMWInst::FMax;
}

/// Replace `RMW` with
///
///   loop:  %loaded = phi [ %init, %entry ], [ %prev, %loop ]
///          %new    = llvm.minnum/maxnum(%loaded, %operand)
///          %pair   = cmpxchg ptr, bitcast(%loaded), bitcast(%new)
///          %prev   = bitcast(extractvalue %pair, 0)
///          br (extractvalue %pair, 1), %end, %loop
///
/// and hand the original value, %loaded, back to the users of `RMW`.
static void expandToCmpXchgLoop(AtomicRMWInst *RMW) {
  Type *ValueTy = RMW->getType();
  LLVMContext &Ctx = RMW->getContext();
  Value *Ptr = RMW->getPointerOperand();
  Value *Operand = RMW->getValOperand();

  IntegerType *IntTy =
      IntegerType::get(Ctx, ValueTy->getPrimitiveSizeInBits());

  BasicBlock *EntryBB = RMW->getParent();
  BasicBlock *EndBB = EntryBB->splitBasicBlock(RMW->getIterator(), "atomicrmw.end");
  BasicBlock *LoopBB =
      BasicBlock::Create(Ctx, "atomicrmw.start", EntryBB->getParent(), EndBB);

  // splitBasicBlock left an unconditional branch to EndBB; retarget it.
  EntryBB->getTerminator()->eraseFromParent();
  IRBuilder<> Builder(EntryBB);
  LoadInst *Init = Builder.CreateAlignedLoad(ValueTy, Ptr, RMW->getAlign());
  Init->setAtomic(AtomicOrdering::Monotonic, RMW->getSyncScopeID());
  Init->setVolatile(RMW->isVolatile());
  Builder.CreateBr(LoopBB);

  Builder.SetInsertPoint(LoopBB);
  PHINode *Loaded = Builder.CreatePHI(ValueTy, 2, "loaded");
  Loaded->addIncoming(Init, EntryBB);

  Intrinsic::ID ID = RMW->getOperation() == AtomicRMWInst::FMin
                         ? Intrinsic::minnum
                         : Intrinsic::maxnum;
  Value *New = Builder.CreateBinaryIntrinsic(ID, Loaded, Operand);

  // cmpxchg only accepts integer and pointer types.
  Value *LoadedInt = Builder.CreateBitCast(Loaded, IntTy);
  Value *NewInt = Builder.CreateBitCast(New, IntTy);
  AtomicOrdering Success = RMW->getOrdering();
  AtomicOrdering Failure =
      AtomicCmpXchgInst::getStrongestFailureOrdering(Success);
  AtomicCmpXchgInst *Pair =
      Builder.CreateAtomicCmpXchg(Ptr, LoadedInt, NewInt, RMW->getAlign(),
                                  Success, Failure, RMW->getSyncScopeID());
  Pair->setVolatile(RMW->isVolatile());
  Pair->setWeak(false);

  Value *PrevInt = Builder.CreateExtractValue(Pair, 0, "prev");
  Value *Done = Builder.CreateExtractValue(Pair, 1, "done");
  Value *Prev = Builder.CreateBitCast(PrevInt, ValueTy);
  Loaded->addIncoming(Prev, LoopBB);
  Builder.CreateCondBr(Done, EndBB, LoopBB);

  RMW->replaceAllUsesWith(Loaded);
  RMW->eraseFromParent();
}

static bool lowerFPAtomicMinMax(Function &F) {
  SmallVector<AtomicRMWInst *, 8> WorkList;
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
        if (isFPMinMax(*RMW))
          WorkList.push_back(RMW);

  for (auto *RMW : WorkList)
    expandToCmpXchgLoop(RMW);

  return !WorkList.empty();
}

PreservedAnalyses HipLowerFPAtomicMinMaxPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  return lowerFPAtomicMinMax(F) ? PreservedAnalyses::none()
                                : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_NAME, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_NAME) {
                    FPM.addPass(HipLowerFPAtomicMinMaxPass());
                    return true;
                  }
                  return false;
                });
          }};
}
