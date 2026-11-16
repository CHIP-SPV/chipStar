//===- RemoveNoInlineOptNoneAttrs.h ---------------------------------------===//
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

#ifndef LLVM_PASSES_REMOVE_NO_INLINE_OPT_NONE_ATTRS_H
#define LLVM_PASSES_REMOVE_NO_INLINE_OPT_NONE_ATTRS_H

#include "PassInfoMixinCompat.h"

using namespace llvm;

// Strip compiler-generated optnone+noinline pairs (debug builds) but
// preserve user-annotated __attribute__((noinline)) so it propagates
// to SPIR-V FunctionControl=DontInline.  When both optnone and
// noinline are present, they came from the compiler; noinline alone
// means the user asked for it explicitly.
class RemoveNoInlineOptNoneAttrsPass
    : public HipRequiredPassInfoMixin<RemoveNoInlineOptNoneAttrsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
