//===- HipLowerHintIntrinsics.cpp -----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Erases or folds intrinsics llvm-spirv rejects, to values LangRef permits.
//
// WORKAROUND(CHIP-SPV/chipStar#1633, KhronosGroup/SPIRV-LLVM-Translator#3990):
// llvm-spirv rejects these intrinsics. Remove each case once every supported
// SPIR-V producer accepts it.
//
// Copyright (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerHintIntrinsics.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

/// Rewrite one call. Returns true if \p II was replaced and erased.
static bool lowerCall(IntrinsicInst *II, const DataLayout &DL) {
  Type *Ty = II->getType();
  // Null means the call produces no value and is simply dropped.
  Value *Repl = nullptr;

  switch (II->getIntrinsicID()) {
  case Intrinsic::prefetch:
    break;
  case Intrinsic::readcyclecounter:
  case Intrinsic::readsteadycounter:
    Repl = ConstantInt::get(Ty, 0);
    break;
  case Intrinsic::get_rounding:
    // The default OpenCL rounding mode is nearest even, FLT_ROUNDS encoding 1.
    Repl = ConstantInt::get(Ty, 1);
    break;
  case Intrinsic::allow_runtime_check:
    // Clang's default when no -lower-allow-check option is given.
    Repl = ConstantInt::getTrue(II->getContext());
    break;
  case Intrinsic::returnaddress:
  case Intrinsic::frameaddress:
    Repl = ConstantPointerNull::get(cast<PointerType>(Ty));
    break;
  case Intrinsic::objectsize:
    // No TLI: it only adds recognition of host libc allocators.
    Repl = lowerObjectSizeCall(II, DL, /*TLI=*/nullptr, /*MustSucceed=*/true);
    break;
  case Intrinsic::memcpy_inline: {
    // OpCopyMemorySized must not have a constant zero Size.
    auto *MC = cast<MemCpyInst>(II);
    auto *Len = dyn_cast<ConstantInt>(MC->getLength());
    if (!Len || !Len->isZero())
      IRBuilder<>(MC).CreateMemCpy(MC->getRawDest(), MC->getDestAlign(),
                                   MC->getRawSource(), MC->getSourceAlign(),
                                   MC->getLength(), MC->isVolatile());
    break;
  }
  default:
    return false;
  }

  if (Repl)
    II->replaceAllUsesWith(Repl);
  II->eraseFromParent();
  return true;
}

} // namespace

PreservedAnalyses HipLowerHintIntrinsicsPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  SmallVector<IntrinsicInst *, 8> Worklist;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *II = dyn_cast<IntrinsicInst>(&I))
        Worklist.push_back(II);

  bool Changed = false;
  for (IntrinsicInst *II : Worklist)
    Changed |= lowerCall(II, M.getDataLayout());

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
