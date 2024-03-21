//===- HipIGBADetector.cpp --------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass analyzes module to find out if it has potential indirect global
// buffer accesses (IGBA). The outcome of the analysis is stored in a magic
// variable for the chipStar runtime:
//
//   uint8_t __chip_module_has_no_IGBAs = <result>;
//
// Where the result is one if the are no potential IGBAs and otherwise it is
// zero.
//
// If there would be an IGPA in the module, there has to to be a load
// instruction with a pointer operand which is either loaded from memory or
// crafted from an integer (which OTOH is loaded from somewhere else). The
// analysis is very simple and naive: we look for pointer load and inttoptr
// instructions in the whole module. If we see any, we conclude there are
// potential IGBAs. Downsides of this are that
//
// * may-have-IGBAs is concluded even tough only one kernel has IGBAs an
//   others don't
//
// * unoptimized modules (-O0) will likely likely result in may-have-IGBAs
// * conclusion
//
// The motivation for this analysis is to reduce clSetKernelExecInfo() calls in
// the OpenCL backend.
//
// (c) 2024 Henry Linjamäki / Intel
//===----------------------------------------------------------------------===//
#include "HipIGBADetector.h"

#include "LLVMSPIRV.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#define PASS_NAME "hip-igpa-detector"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

static bool hasPotentialIGBAs(Module &M) {
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        if (isa<IntToPtrInst>(&I))
          return true;
        if (auto *LI = dyn_cast<LoadInst>(&I))
          return LI->getType()->isPointerTy();
      }
  return false;
}

static bool detectIGBAs(Module &M) {
  constexpr auto *MagicVarName = "__chip_module_has_no_IGBAs";

  if (M.getGlobalVariable(MagicVarName))
    return false; // Bail out: the module has already been processed.

  bool Result = hasPotentialIGBAs(M);
  LLVM_DEBUG(dbgs() << "Has IGBAs: " << Result << "\n");

  auto *Init = ConstantInt::get(IntegerType::get(M.getContext(), 8), !Result);
  (void)new GlobalVariable(
      M, Init->getType(), true,
      // Mark the GV as external for keeping it alive at least until the
      // chipStar runtime reads it.
      GlobalValue::ExternalLinkage, Init, MagicVarName, nullptr,
      GlobalValue::NotThreadLocal /* Default value */,
      // Global-scope variables may not have Function storage class.
      // TODO: use private storage class?
      SPIRV_CROSSWORKGROUP_AS);

  return true;
}

PreservedAnalyses HipIGBADetectorPass::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  return detectIGBAs(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_NAME, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_NAME) {
                    MPM.addPass(HipIGBADetectorPass());
                    return true;
                  }
                  return false;
                });
          }};
}
