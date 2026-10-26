//===- HipLowerMemset.h ---------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Consumers that rebuild LLVM IR from SPIR-V with an older LLVM (e.g. Arm Mali,
// PoCL) reject newer llvm.memset name mangling (SPIRV-LLVM-Translator#2128);
// expanding memsets keeps them out.
//
// (c) 2023 Henry Linjamäki / Intel
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_MEMSET_H
#define LLVM_PASSES_HIP_LOWER_MEMSET_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

class HipLowerMemsetPass : public HipRequiredPassInfoMixin<HipLowerMemsetPass> {
public:
  PreservedAnalyses run(Function &M, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
