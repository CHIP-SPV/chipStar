//===- HipFPContract.cpp --------------------------------------------------===//
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

#include "HipFPContract.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

using namespace llvm;

#ifdef CHIP_LLVM_USE_INTERGRATED_SPIRV
namespace {
// What makes llvm-spirv disable contraction for the enclosing function.
static bool forbidsContraction(const Instruction &I) {
  if (auto *B = dyn_cast<BinaryOperator>(&I))
    return (B->getOpcode() == Instruction::FAdd ||
            B->getOpcode() == Instruction::FSub) &&
           !B->hasAllowContract();
  auto *CI = dyn_cast<CallInst>(&I);
  if (!CI || isa<IntrinsicInst>(CI))
    return false;
  const Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return true;
  if (!Callee->isDeclaration())
    return false;
  // Builtins are named printf, __spirv_*, or mangled as _Z<len><name>.
  StringRef Name = Callee->getName();
  if (Name.size() > 2 && Name.starts_with("_Z") && isDigit(Name[2]))
    Name = Name.drop_front(2).ltrim("0123456789");
  else if (Name != "printf" && !Name.starts_with("__spirv_"))
    return true;
  // Any other __ name is a chipStar runtime helper, not a builtin.
  return Name.starts_with("__") && !Name.starts_with("__spirv_");
}
} // namespace

PreservedAnalyses HipFPContractPass::run(Module &M,
                                         ModuleAnalysisManager &AM) {
  for (const Function &F : M)
    for (const Instruction &I : instructions(F))
      if (forbidsContraction(I))
        return PreservedAnalyses::all();
  M.getOrInsertNamedMetadata("opencl.enable.FP_CONTRACT");
  return PreservedAnalyses::all();
}
#endif
