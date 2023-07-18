//===- HipLowerSwitch.h ---------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lowers switch instructions to be suitable for llvm-spirv.
//
// (c) 2023 chipStar Developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_SWITCH_H
#define LLVM_PASSES_HIP_LOWER_SWITCH_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

class HipLowerSwitchPass : public PassInfoMixin<HipLowerSwitchPass> {
public:
  PreservedAnalyses run(Function &M, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
