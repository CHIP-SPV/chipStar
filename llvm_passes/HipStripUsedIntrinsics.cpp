//===- HipStripUsedIntrinsics.cpp -----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM pass to remove llvm.used and llvm.compiler.used intrinsic variables (see
// LLVM lang ref for details) from HIP device code modules.
//
// LLVM/Clang generates these in some cases.  We remove them because they also
// generate illegal address space casts (when the IR is passed to llvm-spirv
// translator tool).  It is safe to remove llvm(.compiler).used variables as
// they are meaningless for SPIR-V.
//
// This pass is likely not needed when SPIR-V backend land on LLVM in the
// future.
//
// (c) 2021-2022 Pekka Jääskeläinen / Parmance for Argonne National Laboratory
//===----------------------------------------------------------------------===//


#include "HipStripUsedIntrinsics.h"

#if LLVM_VERSION_MAJOR >= 14
#include "llvm/Pass.h"
#endif

using namespace llvm;

namespace {

bool stripGlobalVar(Module &M, StringRef GVName) {
  auto *GV = M.getGlobalVariable(GVName);
  if (!GV)
    return false;
  assert(GV->getNumUses() == 0 && "GV has uses.");
  GV->eraseFromParent();
  return true;
}

class HipStripCompilerUsedLegacyPass : public ModulePass {
public:
  static char ID;
  HipStripCompilerUsedLegacyPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    bool Changed = stripGlobalVar(M, "llvm.used");
    Changed |= stripGlobalVar(M, "llvm.compiler.used");
    return Changed;
  }

  StringRef getPassName() const override {
    return "Strip llvm.compiler.used variables.";
  }
};

} // namespace

char HipStripCompilerUsedLegacyPass::ID = 0;
static RegisterPass<HipStripCompilerUsedLegacyPass>
    X("hip-strip-compiler-used", "Strip llvm.compiler.used variables.");

// Pass hook for the new pass manager.
#if LLVM_VERSION_MAJOR > 11
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

PreservedAnalyses HipStripUsedIntrinsicsPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  bool Changed = stripGlobalVar(M, "llvm.used");
  Changed |= stripGlobalVar(M, "llvm.compiler.used");
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-strip-compiler-used",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-strip-compiler-used") {
                    FPM.addPass(HipStripUsedIntrinsicsPass());
                    return true;
                  }
                  return false;
                });
          }};
}

#endif // LLVM_VERSION_MAJOR > 11
