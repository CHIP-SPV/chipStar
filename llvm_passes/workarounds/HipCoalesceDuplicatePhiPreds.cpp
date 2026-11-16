//===- HipCoalesceDuplicatePhiPreds.cpp -----------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Copyright (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipCoalesceDuplicatePhiPreds.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

PreservedAnalyses
HipCoalesceDuplicatePhiPredsPass::run(Function &F, FunctionAnalysisManager &) {
  SmallVector<BasicBlock *, 16> PhiBlocks;
  for (BasicBlock &BB : F)
    if (isa<PHINode>(BB.begin()))
      PhiBlocks.push_back(&BB);

  bool Changed = false;
  for (BasicBlock *BB : PhiBlocks) {
    SmallSetVector<BasicBlock *, 8> Preds(pred_begin(BB), pred_end(BB));
    for (BasicBlock *Pred : Preds) {
      if (count(successors(Pred), BB) < 2)
        continue;
      BasicBlock *Fwd = BasicBlock::Create(F.getContext(),
                                           BB->getName() + ".dedup", &F, BB);
      IRBuilder<>(Fwd).CreateBr(BB);
      Instruction *TI = Pred->getTerminator();
      for (unsigned I = 0, E = TI->getNumSuccessors(); I != E; ++I)
        if (TI->getSuccessor(I) == BB)
          TI->setSuccessor(I, Fwd);
      // The verifier guarantees every entry from Pred carries the same value.
      for (PHINode &Phi : BB->phis()) {
        Value *V = Phi.getIncomingValueForBlock(Pred);
        for (int Idx; (Idx = Phi.getBasicBlockIndex(Pred)) >= 0;)
          Phi.removeIncomingValue(Idx, false);
        Phi.addIncoming(V, Fwd);
      }
      Changed = true;
    }
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
