//===- HipStripDebugInfo.h ------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM pass removing debug information from HIP device code modules.
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_STRIP_DEBUG_INFO_H
#define LLVM_PASSES_HIP_STRIP_DEBUG_INFO_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

class HipStripDebugInfoPass : public PassInfoMixin<HipStripDebugInfoPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
