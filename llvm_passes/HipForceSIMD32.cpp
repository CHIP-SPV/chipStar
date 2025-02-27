//===- HipForceSIMD32.cpp -------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR pass to force all kernels to use SIMD32 (subgroup size of 32).
//
// (c) 2025 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipForceSIMD32.h"

#include <llvm/IR/Metadata.h>
#include "llvm/IR/Module.h"
#include <llvm/IR/Constants.h>

#define DEBUG_TYPE "hip-force-simd32"

PreservedAnalyses HipForceSIMD32Pass::run(Module &Mod, ModuleAnalysisManager &AM) {
  auto &Ctx = Mod.getContext();
  bool Modified = false;
  
  for (auto &F : Mod) {
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    IntegerType *I32Type = IntegerType::get(Ctx, 32);
    F.setMetadata("intel_reqd_sub_group_size",
                  MDNode::get(Ctx, ConstantAsMetadata::get(ConstantInt::get(
                                       I32Type, 32))));
    Modified = true;
  }

  // The metadata should not impact other chipStar passes.
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// Pass hook for the new pass manager.
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-force-simd32", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-force-simd32") {
                    MPM.addPass(HipForceSIMD32Pass());
                    return true;
                  }
                  return false;
                });
          }};
}
