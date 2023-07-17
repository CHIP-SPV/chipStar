//===- HipDefrost.h -------------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Removes freeze instructions from code for SPIRV-LLVM translator tool.
//
// (c) 2021 Paulius Velesko for Argonne National Laboratory
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_DEFROST_H
#define LLVM_PASSES_HIP_DEFROST_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR > 11
class HipDefrostPass
    : public PassInfoMixin<HipDefrostPass> {
public:
  PreservedAnalyses run(Function &M, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif

#endif
