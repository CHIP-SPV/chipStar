//===- HipLowerHintIntrinsics.h -------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Erases or constant folds the LLVM intrinsics that carry no observable
// meaning on a SPIR-V device, so that none of them reaches SPIR-V emission.
//
// Copyright (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_HINT_INTRINSICS_H
#define LLVM_PASSES_HIP_LOWER_HINT_INTRINSICS_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

class HipLowerHintIntrinsicsPass
    : public PassInfoMixin<HipLowerHintIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
