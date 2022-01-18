//===- HipPrintf.cppp -----------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR pass to convert calls to the CUDA/HIP printf() to OpenCL/SPIR-V
// compatible ones.
//
// (c) 2021-2022 Pekka Jääskeläinen / Parmance for Argonne National Laboratory
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_PRINTF_H
#define LLVM_PASSES_HIP_PRINTF_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"

#include <map>

using namespace llvm;

class HipPrintfToOpenCLPrintfPass
    : public PassInfoMixin<HipPrintfToOpenCLPrintfPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  Constant *getOrCreateStrLiteralArg(
      const std::string& Str, llvm::IRBuilder<>& B);
  Function *getOrCreatePrintStringF();
  Value* cloneStrArgToConstantAS(
    Value *StrArg, llvm::IRBuilder<>& B, bool *IsEmpty);

  std::map<std::string, Constant*> LiteralArgs_;
  Module *M_;
};

#endif
