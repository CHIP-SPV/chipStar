//===- HipCleanup.h -------------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR pass to clean up chipStar/HIP internal globals and stub functions
// that reference them. Removes __hip_cuid*, __hip_fatbin*, non-var __chip_*
// globals and stubs their users.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_CLEANUP_H
#define LLVM_PASSES_HIP_CLEANUP_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

class HipCleanupPass : public PassInfoMixin<HipCleanupPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
