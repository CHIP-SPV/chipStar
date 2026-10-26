//===- HipLowerZeroLengthArrays.h -----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lowers occurrences of zero length array types which are not supported by
// llvm-spirv.
//
// WORKAROUND(CHIP-SPV/chipStar#2109, llvm/llvm-project#172367): the LLVM 21
// SPIR-V backend emits [0 x T] as OpTypeRuntimeArray, which IGC rejects. Remove
// when LLVM 21 support is dropped.
//
// Copyright (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_ZERO_LENGTH_ARRAYS_H
#define LLVM_PASSES_HIP_LOWER_ZERO_LENGTH_ARRAYS_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR < 14
#error LLVM 14+ required.
#endif

class HipLowerZeroLengthArraysPass
    : public HipRequiredPassInfoMixin<HipLowerZeroLengthArraysPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
