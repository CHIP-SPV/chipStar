#ifndef LLVM_PASSES_HIP_GLOBAL_VARIABLES_H
#define LLVM_PASSES_HIP_GLOBAL_VARIABLES_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR > 11
class HipGlobalVariablesPass : public PassInfoMixin<HipGlobalVariablesPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
#endif

#endif
