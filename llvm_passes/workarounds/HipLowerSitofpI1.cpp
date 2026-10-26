//===- HipLowerSitofpI1.cpp -----------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lowers sitofp i1 to a select.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerSitofpI1.h"

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

using namespace llvm;

PreservedAnalyses HipLowerSitofpI1Pass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
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
