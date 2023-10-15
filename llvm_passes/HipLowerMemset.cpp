//===- HipLowerMemset.cpp -------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expand llvm.memset intrinsics to loops before handing the bitcode to
// llvm-spirv translator. The translator expands memset intrinsics to an
// emulation function with a name encoding the destination pointer type
// (e.g. "spirv.spirv.llvm_memset_p0_i32..."). During reverse translation,
// llvm-spirv attempts to recreate the original memset intrinsic which can be
// incorrect if we are jumping from opaque pointer world to typed pointer world
// or vice versa. LLVM module verification may fail with the following message:
//
//   Fails to verify module: Intrinsic name not mangled correctly for
//   type arguments!
//
// This issue has been filed here:
// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/2128
//
// llvm.memcpy does not have this issue as it gets translated to actual SPIR-V
// instruction (OpCopyMemorySized).
//
// (c) 2023 Henry Linjamäki / Intel
//===----------------------------------------------------------------------===//
#include "HipLowerMemset.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
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

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-lower-memset", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-lower-memset") {
                    FPM.addPass(HipLowerMemsetPass());
                    return true;
                  }
                  return false;
                });
          }};
}
