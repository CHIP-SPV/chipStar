//===- HipIGBADetector.h ----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// (c) 2024 Henry Linjamäki / Intel
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_IGBA_DETECTOR_H
#define LLVM_PASSES_HIP_IGBA_DETECTOR_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

class HipIGBADetectorPass
    : public HipRequiredPassInfoMixin<HipIGBADetectorPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
