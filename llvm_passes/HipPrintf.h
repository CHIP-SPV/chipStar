#ifndef LLVM_PASSES_HIP_PRINTF_H
#define LLVM_PASSES_HIP_PRINTF_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#define SPIRV_OPENCL_PRINTF_FMT_ARG_AS 2

#if LLVM_VERSION_MAJOR > 11
class HipPrintfToOpenCLPrintfPass
    : public PassInfoMixin<HipPrintfToOpenCLPrintfPass> {
public:
  PreservedAnalyses run(Function &M, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif

#endif
