//===- HipLowerSitofpI1.h -------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lowers sitofp i1 to a select.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_SITOFP_I1_H
#define LLVM_PASSES_HIP_LOWER_SITOFP_I1_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

// WORKAROUND(CHIP-SPV/chipStar#1891, KhronosGroup/SPIRV-LLVM-Translator#3918): translator lowers sitofp i1 true to +1.0. Remove when the fix is in the translator release branch chipStar builds.
// WORKAROUND(CHIP-SPV/chipStar#1891, llvm/llvm-project#209232): the in-tree
// backend does the same. Remove when the fix is in the LLVM release chipStar
// builds. Delete the pass only once CanarySitofpI1Translator and
// CanarySitofpI1Backend both fire for every supported LLVM version.
// Rewrites sitofp i1 %c into select %c, -1.0, 0.0.
class HipLowerSitofpI1Pass
    : public HipRequiredPassInfoMixin<HipLowerSitofpI1Pass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
