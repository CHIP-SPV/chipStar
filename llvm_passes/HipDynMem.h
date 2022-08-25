//===- HipDynMem.h --------------------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM Pass to replace dynamically sized shared arrays ("extern __shared__ type[]")
// with a function argument. This is required because CUDA/HIP use a "magic variable"
// for dynamically sized shared memory, while OpenCL API uses a kernel argument
//
// (c) 2021 Paulius Velesko for Argonne National Laboratory
// (c) 2020 Michal Babej for TUNI
// (c) 2022 Michal Babej for Argonne National Laboratory
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_DYN_MEM_H
#define LLVM_PASSES_HIP_DYN_MEM_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR > 11
class HipDynMemExternReplaceNewPass
    : public PassInfoMixin<HipDynMemExternReplaceNewPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif

#endif
