//===- HipLowerMemset.cpp -------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expand llvm.memset to loops: consumers that rebuild LLVM IR from SPIR-V with an
// older LLVM (e.g. Arm Mali, PoCL) reject newer llvm.memset name mangling
// (SPIRV-LLVM-Translator#2128).
//
// (c) 2023 Henry Linjamäki / Intel
//===----------------------------------------------------------------------===//
#include "HipLowerMemset.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Transforms/Utils/LowerMemIntrinsics.h>

using namespace llvm;

static bool lowerMemsets(Function &F) {
  SmallPtrSet<MemSetInst *, 8> WorkList;
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *MemSet = dyn_cast<MemSetInst>(&I))
        WorkList.insert(MemSet);

  for (auto *MemSet : WorkList) {
    expandMemSetAsLoop(MemSet);
    MemSet->eraseFromParent();
  }

  return !WorkList.empty();
}

PreservedAnalyses HipLowerMemsetPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  return lowerMemsets(F) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
