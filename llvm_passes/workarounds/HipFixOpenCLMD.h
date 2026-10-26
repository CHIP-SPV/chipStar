//===- HipFixOpenCLMD.h ---------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Sets the OpenCL version metadata to 2.0.
//
// WORKAROUND(CHIP-SPV/chipStar#2101, no upstream report): clang tags HIP device
// IR with opencl.ocl.version 0.0 and llvm-spirv aborts on the mismatch with the
// CL2.0 device library. Remove when either side stops doing that.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_FIX_OPEN_CLMD_H
#define LLVM_PASSES_HIP_FIX_OPEN_CLMD_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

class HipFixOpenCLMDPass : public HipRequiredPassInfoMixin<HipFixOpenCLMDPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
