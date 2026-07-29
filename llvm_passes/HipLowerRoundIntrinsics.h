//===- HipLowerRoundIntrinsics.h ------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expands llvm.llround.* and llvm.llrint.*, which llvm-spirv cannot translate,
// to llvm.round / llvm.rint plus a conversion.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_ROUND_INTRINSICS_H
#define LLVM_PASSES_HIP_LOWER_ROUND_INTRINSICS_H

#include <llvm/IR/PassManager.h>

using namespace llvm;

class HipLowerRoundIntrinsicsPass
    : public PassInfoMixin<HipLowerRoundIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
