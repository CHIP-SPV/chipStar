//===- HipLowerFPAtomicMinMax.h -------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expands `atomicrmw fmin` and `atomicrmw fmax` to cmpxchg loops so the module
// does not require SPV_EXT_shader_atomic_float_min_max.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_FP_ATOMIC_MIN_MAX_H
#define LLVM_PASSES_HIP_LOWER_FP_ATOMIC_MIN_MAX_H

#include <llvm/IR/PassManager.h>

using namespace llvm;

class HipLowerFPAtomicMinMaxPass
    : public PassInfoMixin<HipLowerFPAtomicMinMaxPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
