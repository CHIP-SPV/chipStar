//===- HipCoalesceDuplicatePhiPreds.h -------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Routes every group of edges from one block into the same phi block through a
// single forwarding block, so no phi lists a predecessor more than once.
//
// Copyright (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_COALESCE_DUPLICATE_PHI_PREDS_H
#define LLVM_PASSES_HIP_COALESCE_DUPLICATE_PHI_PREDS_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

// WORKAROUND(CHIP-SPV/chipStar#1680, KhronosGroup/SPIRV-LLVM-Translator#3866): llvm-spirv emits one OpPhi entry per LLVM phi entry, duplicating predecessors. Remove when the pinned llvm_release branch includes #3866.
class HipCoalesceDuplicatePhiPredsPass
    : public PassInfoMixin<HipCoalesceDuplicatePhiPredsPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
