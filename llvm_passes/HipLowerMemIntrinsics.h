//===- HipLowerMemIntrinsics.h --------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expands llvm.memset.* to a loop to work around an issue in llvm-spirv reverse
// translation, and llvm.memcpy.inline.* to a loop because no SPIR-V producer
// accepts it.
//
// (c) 2023 Henry Linjamäki / Intel
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_MEM_INTRINSICS_H
#define LLVM_PASSES_HIP_LOWER_MEM_INTRINSICS_H

#include <llvm/IR/PassManager.h>

using namespace llvm;

class HipLowerMemIntrinsicsPass
    : public PassInfoMixin<HipLowerMemIntrinsicsPass> {
public:
  PreservedAnalyses run(Function &M, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
