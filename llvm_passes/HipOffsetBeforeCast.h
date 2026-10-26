//===- HipOffsetBeforeCast.h ----------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Rewrites a GEP over an addrspacecast of a global as a cast of the GEP.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_OFFSET_BEFORE_CAST_H
#define LLVM_PASSES_HIP_OFFSET_BEFORE_CAST_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

// WORKAROUND(CHIP-SPV/chipStar#1693, CHIP-SPV/chipStar#1695; no upstream
// reports): in a global initializer, the in-tree SPIR-V backend aborts on a
// getelementptr over an addrspacecast of a global, and IGC stores 0 for it.
// Rewrites it as an addrspacecast of the getelementptr. Remove when both
// handle the original and #1703 is fixed.
class HipOffsetBeforeCastPass
    : public HipRequiredPassInfoMixin<HipOffsetBeforeCastPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
