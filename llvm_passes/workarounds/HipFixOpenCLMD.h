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
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_FIX_OPEN_CLMD_H
#define LLVM_PASSES_HIP_FIX_OPEN_CLMD_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

// LLVM commit 15a1769631ff0b2b3e830b03e51ae5f54f08a0ab introduces
// 'opencl.ocl.version' module metadata into device code. This triggers an
// assertion in SPIRV-LLVM Translator if the HIP device code and linked device
// bitcode has mixed OpenCL version metadata. The commit in question inserts
// OpenCL version 0.0 in HIP compilation mode (in CUDA mode it is 2.0).
//
// This pass works around the issue until some fix is introduced in Clang. The
// issue is fixed by setting OpenCL version to the same as bitcode library
// (2.0).
class HipFixOpenCLMDPass : public HipRequiredPassInfoMixin<HipFixOpenCLMDPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
