//===- HipFPContract.h ----------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Enables FP contraction for the in-tree SPIR-V backend.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_FPCONTRACT_H
#define LLVM_PASSES_HIP_FPCONTRACT_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

#ifdef CHIP_LLVM_USE_INTERGRATED_SPIRV
// WORKAROUND(CHIP-SPV/chipStar#1654, llvm/llvm-project#206404): the in-tree
// backend puts ContractionOff on every kernel unless opencl.enable.FP_CONTRACT
// is present, llvm-spirv only on kernels reaching an op that forbids
// contraction. Remove when the backend's default decides from the operations.
class HipFPContractPass : public HipRequiredPassInfoMixin<HipFPContractPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif

#endif
