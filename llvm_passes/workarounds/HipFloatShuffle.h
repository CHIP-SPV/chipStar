//===- HipFloatShuffle.h --------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Turns integer views of floats in sub-group shuffles into float shuffles.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_FLOAT_SHUFFLE_H
#define LLVM_PASSES_HIP_FLOAT_SHUFFLE_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

// WORKAROUND(CHIP-SPV/chipStar#1896, intel/intel-graphics-compiler (not yet reported)): IGC aliases int views of a float in SIMD32 shuffles. Remove when IGC fixes emitSimdShuffle.
// Rewrites sub_group_shuffle{,_xor}(bitcast float %f to i32) as the float
// shuffle of %f.
class HipFloatShufflePass
    : public HipRequiredPassInfoMixin<HipFloatShufflePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
