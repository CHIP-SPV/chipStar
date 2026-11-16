//===- HipTextureLowering.cpp ---------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A pass to lower HIP texture functions.
//
// (c) 2022 Henry Linjamäki / Parmance for Argonne National Laboratory
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_TEXTURE_NEW_H
#define LLVM_PASSES_HIP_TEXTURE_NEW_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

class HipTextureLoweringPass
    : public HipRequiredPassInfoMixin<HipTextureLoweringPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
