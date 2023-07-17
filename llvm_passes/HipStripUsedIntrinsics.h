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
// (c) 2021-2022 Pekka Jääskeläinen / Parmance for Argonne National Laboratory
//===----------------------------------------------------------------------===//


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
