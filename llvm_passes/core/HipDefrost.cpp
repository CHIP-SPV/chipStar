//===- HipDefrost.cpp -----------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Removes freeze instructions.
//
// HipShuffleShift cannot see through freeze in a shuffle index; dropping freeze
// here keeps constant-offset shuffles rewritable.
//
// (c) 2021 Paulius Velesko for Argonne National Laboratory
//===----------------------------------------------------------------------===//


#include "HipDefrost.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {

bool defrost(Function &F) {
  SmallPtrSet<Instruction *, 8> EraseList;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!isa<FreezeInst>(I))
        continue;
      I.replaceAllUsesWith(I.getOperand(0));
      EraseList.insert(&I);
    }
  }
  for (auto I : EraseList)
    I->eraseFromParent();
  return EraseList.size();
}

class HipDefrostLegacyPass : public FunctionPass {
public:
  static char ID;
  HipDefrostLegacyPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override { return defrost(F); }

  StringRef getPassName() const override {
    return "Remove freeze instructions.";
  }
};

} // namespace

char HipDefrostLegacyPass::ID = 0;
static RegisterPass<HipDefrostLegacyPass> X("hip-defrost",
                                            "Remove freeze instructions.");

// Pass hook for the new pass manager.
#include "llvm/Passes/PassBuilder.h"
#include "PassPluginCompat.h"

PreservedAnalyses HipDefrostPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  return defrost(F) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
