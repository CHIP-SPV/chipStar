//===- HipForceSIMD32.h ---------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR pass to force all kernels to use SIMD32 (subgroup size of 32).
//
// (c) 2025 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_FORCE_SIMD32_H
#define LLVM_PASSES_HIP_FORCE_SIMD32_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

class HipForceSIMD32Pass : public PassInfoMixin<HipForceSIMD32Pass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
