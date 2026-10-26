//===- HipLowerMemmove.cpp ------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lowers llvm.memmove to loops.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerMemmove.h"

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include "llvm/IR/Module.h"

using namespace llvm;

#ifdef CHIP_LLVM_USE_INTERGRATED_SPIRV
PreservedAnalyses HipLowerMemmovePass::run(Function &F,
                                           FunctionAnalysisManager &AM) {
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
#endif
