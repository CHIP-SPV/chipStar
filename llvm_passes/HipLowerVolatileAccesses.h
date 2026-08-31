//===- HipLowerVolatileAccesses.h -----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Marks volatile loads and stores through global and generic pointers
// !nontemporal, which the SPIR-V producers emit as the Nontemporal memory
// operand and IGC turns into an L1 uncached access. SPIR-V's Volatile memory
// operand says nothing about caching, so without this a volatile global access
// is served from a core's L1 and loses the meaning CUDA gives it (PTX
// ld.volatile / st.volatile). The accesses stay non-atomic: see the comment in
// HipLowerVolatileAccesses.cpp for why an atomic form is not an option.
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
