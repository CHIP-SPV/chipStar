//===- HipLowerOverflowIntrinsics.h ---------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expands llvm.{u,s}mul.with.overflow into plain integer arithmetic so that
// neither intrinsic reaches SPIR-V emission.
//
// Copyright (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_OVERFLOW_INTRINSICS_H
#define LLVM_PASSES_HIP_LOWER_OVERFLOW_INTRINSICS_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR < 14
#error LLVM 14+ required.
#endif

class HipLowerOverflowIntrinsicsPass
    : public PassInfoMixin<HipLowerOverflowIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
