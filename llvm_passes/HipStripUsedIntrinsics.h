
#ifndef LLVM_PASSES_HIP_STRIP_COMPILER_USED_H
#define LLVM_PASSES_HIP_STRIP_COMPILER_USED_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR > 11
class HipStripUsedIntrinsicsPass
    : public PassInfoMixin<HipStripUsedIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif

#endif
