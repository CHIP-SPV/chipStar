//===- HipLowerRoundIntrinsics.h ------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Rewrites the math intrinsics the SPIR-V producers have no translation for.
// The type-crossing rounding intrinsics llvm.lround.*, llvm.llround.*,
// llvm.lrint.* and llvm.llrint.* become llvm.round / llvm.rint plus a
// conversion: they return an integer, and OpenCL.std has only float to float
// rounding instructions, so neither SPIR-V producer can translate them
// directly. llvm.ldexp becomes a call to the OpenCL ldexp builtin.
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
