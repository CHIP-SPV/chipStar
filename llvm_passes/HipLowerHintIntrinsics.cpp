//===- HipLowerHintIntrinsics.cpp -----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Erases or constant folds intrinsics that clang emits for ordinary __builtin_*
// calls and that a SPIR-V device cannot express.
//
// Why: neither SPIR-V producer chipStar supports accepts them. The llvm-spirv
// translator aborts translation with
//
//   InvalidFunctionCall: Unexpected llvm intrinsic:
//    llvm.prefetch.p4
//
// and the in-tree SPIR-V backend aborts code generation with
//
//   LLVM ERROR: unable to legalize instruction: G_PREFETCH %22:pid(p4), 0, 3, 1
//
// chipStar puts all device code of a binary in one module, so one
// __builtin_prefetch anywhere in it makes every kernel in the program fail to
// build, and neither diagnostic carries a source location. __builtin_prefetch
// in particular is pervasive in the CPU code that gets ported to HIP.
//
// Each rewrite below is one the LLVM Language Reference explicitly permits, so
// the pass changes no defined behavior:
//
//   llvm.prefetch             erased. "The 'llvm.prefetch' intrinsic is a hint
//                             to the code generator to insert a prefetch
//                             instruction if supported; otherwise, it is a
//                             noop." and "This intrinsic does not modify the
//                             behavior of the program."
//   llvm.readcyclecounter     0. "On backends without support, this is lowered
//   llvm.readsteadycounter    to a constant 0."
//   llvm.get.rounding         1. LangRef fixes the encoding of the result, 1
//                             being "to nearest, ties to even". That is the
//                             only rounding mode an OpenCL device offers: "The
//                             only default floating-point rounding mode
//                             supported is round to nearest even i.e the
//                             default rounding mode will be _rte for
//                             floating-point types." (OpenCL C specification,
//                             Rounding Modes). Guarded by the absence of
//                             llvm.set.rounding so a folded getter can never
//                             contradict a setter; clang rejects
//                             __builtin_set_flt_rounds on spirv64 outright, so
//                             the guard is not expected to fire.
//   llvm.allow.runtime.check  false. "For each evaluation of a call to this
//                             intrinsic, the program must be valid and correct
//                             both if it returns true and if it returns false."
//                             false elides the guarded check, which is the
//                             better choice on a GPU.
//   llvm.returnaddress        null. Both "either return a pointer indicating
//   llvm.frameaddress         the [return|frame] address of the specified call
//                             frame, or zero if it cannot be identified", and a
//                             SPIR-V kernel has no addressable call frame.
//   llvm.objectsize           lowerObjectSizeCall, which is what LLVM's own
//                             LowerConstantIntrinsics pass uses. "The
//                             llvm.objectsize intrinsic is lowered to a value
//                             representing the size of the object concerned. If
//                             the size cannot be determined, llvm.objectsize
//                             returns i32/i64 -1 or 0 (depending on the min
//                             argument)." It never reads memory, which is why
//                             the in-tree backend already copes and only the
//                             translator lane needs this.
//
// LangRef quotations are from llvm/docs/LangRef.md at llvmorg-23.1.0-rc2.
//
// llvm.memcpy.inline is handled separately, in HipLowerMemIntrinsics.cpp, since
// it has a real effect that has to be preserved.
//
// Copyright (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerHintIntrinsics.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "PassPluginCompat.h"

#define PASS_NAME "hip-lower-hint-intrinsics"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

namespace {

static bool isHandled(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::prefetch:
  case Intrinsic::readcyclecounter:
  case Intrinsic::readsteadycounter:
  case Intrinsic::get_rounding:
  case Intrinsic::allow_runtime_check:
  case Intrinsic::returnaddress:
  case Intrinsic::frameaddress:
  case Intrinsic::objectsize:
    return true;
  default:
    return false;
  }
}

/// True if anything in \p M can change the floating point rounding mode.
static bool hasRoundingModeSetter(const Module &M) {
  for (const Function &F : M)
    if (F.getIntrinsicID() == Intrinsic::set_rounding && !F.use_empty())
      return true;
  return false;
}

/// Rewrite one call. Returns true if \p II was replaced and erased.
static bool lowerCall(IntrinsicInst *II, const DataLayout &DL,
                      bool FoldRounding) {
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
    if (!FoldRounding)
      return false;
    Repl = ConstantInt::get(Ty, 1);
    break;
  case Intrinsic::allow_runtime_check:
    Repl = ConstantInt::getFalse(II->getContext());
    break;
  case Intrinsic::returnaddress:
  case Intrinsic::frameaddress:
    Repl = ConstantPointerNull::get(cast<PointerType>(Ty));
    break;
  case Intrinsic::objectsize:
    // MustSucceed makes this return the conservative -1 or 0 that LangRef
    // prescribes when the size is not known. TargetLibraryInfo is deliberately
    // not supplied: it would only add recognition of host libc allocators,
    // which device code linked for spirv64 does not call, and the visitor
    // handles a null one.
    Repl = lowerObjectSizeCall(II, DL, /*TLI=*/nullptr, /*MustSucceed=*/true);
    if (!Repl)
      return false;
    break;
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
        if (isHandled(II->getIntrinsicID()))
          Worklist.push_back(II);

  if (Worklist.empty())
    return PreservedAnalyses::all();

  const bool FoldRounding = !hasRoundingModeSetter(M);
  const DataLayout &DL = M.getDataLayout();

  bool Changed = false;
  for (IntrinsicInst *II : Worklist)
    Changed |= lowerCall(II, DL, FoldRounding);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
