//===- HipLowerPointerVectors.h -------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// WORKAROUND(CHIP-SPV/chipStar#1577): rewrites loads and stores of
// <N x ptr> so no vector-of-pointer type reaches SPIR-V emission.
//
// Remove this pass, its registration and its test once the SPIR-V backend
// legalises <N x ptr> itself, or once LLVM stops forming that type for SPIR-V
// targets. See https://github.com/CHIP-SPV/chipStar/issues/1454.
//
// Copyright (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_POINTER_VECTORS_H
#define LLVM_PASSES_HIP_LOWER_POINTER_VECTORS_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR < 14
#error LLVM 14+ required.
#endif

class HipLowerPointerVectorsPass
    : public PassInfoMixin<HipLowerPointerVectorsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
