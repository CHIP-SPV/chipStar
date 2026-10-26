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
// They are meaningless for SPIR-V, and while they exist Internalize/GlobalDCE
// keep everything they name (e.g. the texture and printf helpers) alive.
//
// (c) 2021-2022 Pekka Jääskeläinen / Parmance for Argonne National Laboratory
//===----------------------------------------------------------------------===//


#include "HipStripUsedIntrinsics.h"

#include "llvm/IR/Module.h"
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
#include "llvm/Passes/PassBuilder.h"
#include "PassPluginCompat.h"

PreservedAnalyses HipStripUsedIntrinsicsPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  bool Changed = stripGlobalVar(M, "llvm.used");
  Changed |= stripGlobalVar(M, "llvm.compiler.used");
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
