//===- HipGlobalVariables.h -----------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// In HIP, __constant__ global scope variables in the device code can be accessed
// via a host API. In OpenCL, constant objects in global scope are immutable.
// This implements the global scope variable HIP API with OpenCL backend.
//
// (c) 2022 Parmance for Argonne National Laboratory
//===----------------------------------------------------------------------===//


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
