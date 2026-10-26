//===- HipShuffleShift.h --------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Rewrites constant-offset sub-group shuffles as shifts.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_SHUFFLE_SHIFT_H
#define LLVM_PASSES_HIP_SHUFFLE_SHIFT_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

// WORKAROUND(CHIP-SPV/chipStar#1978, intel/intel-graphics-compiler: no upstream issue yet): IGC lowers a constant-offset sub_group_shuffle to an a0 indirect gather. Remove when IGC lowers it like a shuffle down.
// Rewrites sub_group_shuffle(v, I) to __chip_sg_shift_<T>(v, I, D) when I is
// proven to be the lane, or the lane plus D, and under a run-time test at the
// function entry otherwise. The rtdevlib defines both per device.
class HipShuffleShiftPass
    : public HipRequiredPassInfoMixin<HipShuffleShiftPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
