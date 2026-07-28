//===- HipFunctionPointerAS.h ---------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Move function pointers held in global variable initializers (C++ vtables in
// practice) into the generic address space so the module can be translated to
// SPIR-V.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_FUNCTION_POINTER_AS_H
#define LLVM_PASSES_HIP_FUNCTION_POINTER_AS_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

class HipFunctionPointerASPass
    : public PassInfoMixin<HipFunctionPointerASPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
