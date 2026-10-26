//===- HipFloatShuffle.cpp ------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Turns integer views of floats in sub-group shuffles into float shuffles.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipFloatShuffle.h"

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include <string>

using namespace llvm;

PreservedAnalyses HipFloatShufflePass::run(Function &F,
                                           FunctionAnalysisManager &AM) {
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
