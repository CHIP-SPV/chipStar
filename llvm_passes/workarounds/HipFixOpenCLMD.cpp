//===- HipFixOpenCLMD.cpp -------------------------------------------------===//
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

#include "HipFixOpenCLMD.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

using namespace llvm;

PreservedAnalyses HipFixOpenCLMDPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  constexpr auto OCLVersionMDName = "opencl.ocl.version";
  if (auto *OCLVersionMD = M.getNamedMetadata(OCLVersionMDName)) {
    auto &Ctx = M.getContext();
    auto *Int32Ty = IntegerType::get(Ctx, 32);
    M.eraseNamedMetadata(OCLVersionMD);
    Metadata *OCLVerElts[] = {
        ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 2)),
        ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 0))};
    OCLVersionMD = M.getOrInsertNamedMetadata(OCLVersionMDName);
    OCLVersionMD->addOperand(MDNode::get(Ctx, OCLVerElts));
  }
  // Altering OpenCL metadata probably does not invalidate any analyses.
  return PreservedAnalyses::all();
}
