// A pass to lower HIP texture functions.
//
// (c) 2022 Henry Linjamäki / Parmance for Argonne National Laboratory
#ifndef LLVM_PASSES_HIP_TEXTURE_NEW_H
#define LLVM_PASSES_HIP_TEXTURE_NEW_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR > 11
class HipTextureLoweringPass : public PassInfoMixin<HipTextureLoweringPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif

#endif
