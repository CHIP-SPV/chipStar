//===- HipSharedAddrLocalInit.h -------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Initializes locals from __shared__ addresses in the function.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_SHARED_ADDR_LOCAL_INIT_H
#define LLVM_PASSES_HIP_SHARED_ADDR_LOCAL_INIT_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

// WORKAROUND(CHIP-SPV/chipStar#1691, llvm/llvm-project#198078): clang hoists a
// local aggregate initializer taking __shared__ addresses into a constant
// global, where IGC reads them as null and PoCL aborts. Remove when a clang fix
// that initializes such locals in the function lands.
class HipSharedAddrLocalInitPass
    : public HipRequiredPassInfoMixin<HipSharedAddrLocalInitPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
