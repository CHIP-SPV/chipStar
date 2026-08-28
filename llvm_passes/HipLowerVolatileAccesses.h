//===- HipLowerVolatileAccesses.h -----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Rewrites volatile 32 and 64 bit loads and stores through global and generic
// pointers into relaxed device-scope atomic ones, because that is the SPIR-V
// access with the semantics CUDA gives a volatile global access (PTX
// ld.volatile / st.volatile), while OpLoad / OpStore with the Volatile memory
// operand is served from L1 like any other access.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_VOLATILE_ACCESSES_H
#define LLVM_PASSES_HIP_LOWER_VOLATILE_ACCESSES_H

#include <llvm/IR/PassManager.h>

using namespace llvm;

class HipLowerVolatileAccessesPass
    : public PassInfoMixin<HipLowerVolatileAccessesPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
