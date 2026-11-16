//===- HipLowerMemmove.h --------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lowers llvm.memmove to loops.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_MEMMOVE_H
#define LLVM_PASSES_HIP_LOWER_MEMMOVE_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

#ifdef CHIP_LLVM_USE_INTERGRATED_SPIRV
// WORKAROUND(CHIP-SPV/chipStar#1958, llvm/llvm-project (not yet reported)): the
// in-tree backend emits llvm.memmove as OpCopyMemorySized, which consumers read
// as memcpy. Remove when the SPIR-V backend lowers memmove itself.
class HipLowerMemmovePass
    : public HipRequiredPassInfoMixin<HipLowerMemmovePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif

#endif
