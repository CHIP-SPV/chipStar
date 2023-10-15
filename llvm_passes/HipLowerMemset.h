//===- HipLowerMemset.h ---------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expands llvm.memset.* to loop to workaround a issue in llvm-spirv reverse
// translation.
//
// (c) 2023 Henry Linjamäki / Intel
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_LOWER_MEMSET_H
#define LLVM_PASSES_HIP_LOWER_MEMSET_H

#include <llvm/IR/PassManager.h>

using namespace llvm;

class HipLowerMemsetPass : public PassInfoMixin<HipLowerMemsetPass> {
public:
  PreservedAnalyses run(Function &M, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
