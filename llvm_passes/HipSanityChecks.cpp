//===- HipSanityChecks.cpp ------------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Does sanity checks on the LLVM IR just before HIP-to-SPIR-V lowering.
//
// (c) 2023 CHIP-SPV developers
//===----------------------------------------------------------------------===//
#include "HipSanityChecks.h"

#include "llvm/IR/Instructions.h"
#include "llvm/ADT/BitVector.h"

using namespace llvm;

namespace {
enum Check { IndirectCall, NumChecks };
}

static void checkCallInst(CallInst *CI, BitVector &CaughtChecks) {
  assert(CI);
  if (!CI->getCalledFunction()) {
    Value *CallValue = CI->getCalledOperand()->stripPointerCasts();
    if (auto *F = dyn_cast<Function>(CallValue)) {
      // A function bitcast between the call function operand and the function.
      // Sunch case appears when multiple TUs (and device library) are linked
      // together and a function declared in both have different return and/or
      // parameter types. E.g.
      //
      //   a.hip: extern "C" __device__ int foo(float x) { ... }
      //   b.hip: extern "C" __device__ int foo(double x);
      //          int bar() { return foo(1.0); }
      //
      // This kind of code is very likely leads to breakage due to potential ABI
      // issues.

      // TODO: Fix the printf signature - skip diagnostic as it works now in all
      //       tested backends & driversdespite the type mismatch.
      if (F->getName() == "printf")
        return;

      dbgs() << "Warning: Function type mismatch between caller and callee!\n"
             << "called func: " << F->getName() << "\n"
             << "caller type: " << *CI->getFunctionType() << "\n"
             << "callee type: " << *F->getFunctionType() << "\n"
             << "Unportable or broken code may be generated!\n";

#ifndef NDEBUG
      // Fail visibly, but only for debug build, so issues like these won't slip
      // in silently only to be detected very much later in an obscure way.
      dbgs() << "Aborting (CHIP-SPV debug build mode policy)\n";
      abort();
#endif
    } else {
      // Actual indirect call? Core SPIR-V does not have modeling for indirect
      // calls.
      if (!CaughtChecks.test(Check::IndirectCall)) { // Warn once per module.
        CaughtChecks.set(Check::IndirectCall);
        dbgs() << "Warning: Indirect calls are not yet supported in CHIP-SPV.\n"
               << "Call origin: " << CI->getParent()->getParent()->getName()
               << "\n";
      }

#ifndef NDEBUG
      dbgs() << "Aborting (CHIP-SPV debug build mode policy)\n";
      abort();
#endif
    }
  }
}

PreservedAnalyses HipSanityChecksPass::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  BitVector CaughtChecks(Check::NumChecks);
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          checkCallInst(CI, CaughtChecks);

  return PreservedAnalyses::all();
}
