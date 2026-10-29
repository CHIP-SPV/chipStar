//===- HipLowerMemset.cpp -------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expand llvm.memset intrinsics except small constant ones to loops
// (those are left to the SPIR-V producer) before handing the bitcode to
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
#include <llvm/TargetParser/Triple.h>
#include "PassPluginCompat.h"
#include "chipStarConfig.hh"
#include <llvm/Transforms/Utils/LowerMemIntrinsics.h>

using namespace llvm;

// WORKAROUND(CHIP-SPV/chipStar#2131, Arm Mali driver: not reported): Mali-G52
// miscompiles the translator's lowering of small constant memsets. Remove when
// no supported Mali driver needs it.
#ifdef CHIP_MALI_GPU_WORKAROUNDS
static constexpr bool ExpandAll = true;
#else
static constexpr bool ExpandAll = false;
#endif

static bool lowerMemsets(Function &F) {
  SmallPtrSet<MemSetInst *, 8> WorkList;
  // Vulkan's logical pointers cannot take a memset; expand them all.
  bool All = ExpandAll || Triple(F.getParent()->getTargetTriple())
                                  .getEnvironmentName() == "shadervulkan";
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *MemSet = dyn_cast<MemSetInst>(&I)) {
        // Small constant memsets are lowered natively by the SPIR-V producer.
        auto *Len = dyn_cast<ConstantInt>(MemSet->getLength());
        if (All || MemSet->getIntrinsicID() != Intrinsic::memset ||
            !isa<ConstantInt>(MemSet->getValue()) || !Len ||
            Len->getValue().ugt(256) || Len->isZero())
          WorkList.insert(MemSet);
      }

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

#ifndef CHIP_COMBINED_PASS_PLUGIN
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
#endif // CHIP_COMBINED_PASS_PLUGIN
