//===- HipCanonicalizeGEP.h -----------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Rewrites single-index `getelementptr [N x i8], ptr %p, i64 %i` into an
// explicitly byte-scaled `getelementptr i8, ptr %p, i64 (%i << log2(N))` so
// the SPIR-V emitters produce a uchar-pointer OpPtrAccessChain instead of a
// bitcast-to-array-of-uchar access chain, which the Intel Graphics Compiler
// miscompiles.
//
// Copyright (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_CANONICALIZE_GEP_H
#define LLVM_PASSES_HIP_CANONICALIZE_GEP_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

class HipCanonicalizeGEPPass : public PassInfoMixin<HipCanonicalizeGEPPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
