//===- HipTaskSync.h ---------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A pass to handle HIP cooperative group synchronization related operations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_TASK_SYNC_H
#define LLVM_PASSES_HIP_TASK_SYNC_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR > 11
class HipTaskSyncPass : public PassInfoMixin<HipTaskSyncPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif

#endif

