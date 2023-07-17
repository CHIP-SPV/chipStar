//===- HipEmitLoweredNames.h ----------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A pass to produce a file for mapping hiprtc name expressions to
// lowered/mangled function and global variables needed to implement
// hiprtcGetLoweredName().
//
// Copyright (c) 2021-22 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_EMIT_LOWERED_NAMES_H
#define LLVM_PASSES_HIP_EMIT_LOWERED_NAMES_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR < 14
#error LLVM 14+ required.
#endif

class HipEmitLoweredNamesPass : public PassInfoMixin<HipEmitLoweredNamesPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
