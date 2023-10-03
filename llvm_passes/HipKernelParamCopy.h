//===- HipKernelParamCopy.h ---------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A pass to copy struct kernel params.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_KERNEL_PARAM_COPY_H
#define LLVM_PASSES_HIP_KERNEL_PARAM_COPY_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR > 11
class HipKernelParamCopyPass : public PassInfoMixin<HipKernelParamCopyPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif

#endif
