//===- HipStripDebugInfo.h ------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM pass removing debug information from HIP device code modules.
//
// WORKAROUND(CHIP-SPV/chipStar#1420, no upstream report): llvm-spirv emits a
// forward-referenced DebugTypeComposite cycle that spirv-val rejects. Remove
// when the translator output validates and loads on every consumer.
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_STRIP_DEBUG_INFO_H
#define LLVM_PASSES_HIP_STRIP_DEBUG_INFO_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

class HipStripDebugInfoPass
    : public HipRequiredPassInfoMixin<HipStripDebugInfoPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
