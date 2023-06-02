//===- HipSanityChecks.h --------------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Does sanity checks on the LLVM IR just before HIP-to-SPIR-V lowering.
//
// (c) 2023 CHIP-SPV developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_SANITYCHECKS_H
#define LLVM_PASSES_HIP_SANITYCHECKS_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

class HipSanityChecksPass : public PassInfoMixin<HipSanityChecksPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif // LLVM_PASSES_HIP_SANITYCHECKS_H
