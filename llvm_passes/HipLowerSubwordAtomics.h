//===- HipLowerSubwordAtomics.h -------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Rewrites 8 and 16 bit atomic load, store, atomicrmw and cmpxchg onto the
// aligned 32 bit word that contains the value, because OpenCL SPIR-V consumers
// only implement 32 and 64 bit atomics.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_SUBWORD_ATOMICS_H
#define LLVM_PASSES_HIP_LOWER_SUBWORD_ATOMICS_H

#include <llvm/IR/PassManager.h>

using namespace llvm;

class HipLowerSubwordAtomicsPass
    : public PassInfoMixin<HipLowerSubwordAtomicsPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
