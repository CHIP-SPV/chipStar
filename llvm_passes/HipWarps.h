//===- HipWarps.cpp -------------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR pass to handle warp-width sensitive kernels.
//
// (c) 2022 Pekka Jääskeläinen / Intel
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_WARPS_H
#define LLVM_PASSES_HIP_WARPS_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

class HipWarpsPass : public PassInfoMixin<HipWarpsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
