//===- RemoveNoInlineOptNoneAttrs.cpp -------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Removes compiler-generated optnone+noinline attribute pairs.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#include "RemoveNoInlineOptNoneAttrs.h"

#include "llvm/IR/Module.h"

using namespace llvm;

PreservedAnalyses RemoveNoInlineOptNoneAttrsPass::run(Module &M,
                                                      ModuleAnalysisManager &AM) {
  for (auto &F : M) {
    if (F.hasFnAttribute(Attribute::OptimizeNone)) {
      F.removeFnAttr(Attribute::NoInline);
      F.removeFnAttr(Attribute::OptimizeNone);
    }
  }
  return PreservedAnalyses::none();
}
