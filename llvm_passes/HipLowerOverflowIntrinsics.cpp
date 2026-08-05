//===- HipLowerOverflowIntrinsics.cpp -------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Expands llvm.umul.with.overflow / llvm.smul.with.overflow into plain
// integer arithmetic.
//
// Why: clang emits llvm.umul.with.overflow.i64 for the element-count times
// element-size computation of a C++ array-new expression ("new T[n]"), so any
// device code that news an array hits it. The LLVM SPIR-V producer cannot
// express the intrinsic directly, so it emits a helper function named
// spirv.llvm_umul_with_overflow_i64 and calls that. IGC then reads the SPIR-V
// back into LLVM IR, maps the helper back onto the intrinsic, renames the
// original declaration to "old_llvm.umul.with.overflow.i64" and never
// produces a body for it -- the module build fails with
//
//   error: undefined reference to `old_llvm.umul.with.overflow.i64'
//
// and, because chipStar puts all of a binary's device code in one module,
// that single unresolved symbol takes down every kernel in the program.
//
// Expanding the intrinsic here means the construct never reaches SPIR-V.
// The overflow predicate is computed with a division rather than a 128-bit
// multiply because i128 is not portable across the SPIR-V consumers chipStar
// targets. These intrinsics come from allocation-size checks, which are not
// hot code.
//
// Copyright (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerOverflowIntrinsics.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "PassPluginCompat.h"
#include "llvm/Support/Debug.h"

#define PASS_NAME "hip-lower-overflow-intrinsics"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

namespace {

/// Replace one *.with.overflow call with the equivalent plain arithmetic.
///
/// Unsigned:  lo = a * b;  ov = a != 0 && b > (UMAX / a)
/// Signed:    lo = a * b;  ov = (a == -1 && b == INT_MIN)
///                            || (a != 0 && a != -1 && (lo / a) != b)
///
/// The divisor is fed through a select so the division is never by zero and
/// never sdiv INT_MIN by -1, on any path, including the ones whose result is
/// discarded.
static bool expandCall(IntrinsicInst *II) {
  IRBuilder<> B(II);
  Value *A = II->getArgOperand(0);
  Value *Bv = II->getArgOperand(1);
  Type *Ty = A->getType();
  if (!Ty->isIntegerTy())
    return false;

  Value *Lo = nullptr;
  Value *Ov = nullptr;

  if (II->getIntrinsicID() == Intrinsic::umul_with_overflow) {
    Lo = B.CreateMul(A, Bv, "umulov.lo");
    Constant *Zero = ConstantInt::get(Ty, 0);
    Constant *One = ConstantInt::get(Ty, 1);
    Constant *UMax = Constant::getAllOnesValue(Ty);
    Value *ANotZero = B.CreateICmpNE(A, Zero, "umulov.anz");
    Value *SafeA = B.CreateSelect(ANotZero, A, One, "umulov.safea");
    Value *Limit = B.CreateUDiv(UMax, SafeA, "umulov.limit");
    Value *Over = B.CreateICmpUGT(Bv, Limit, "umulov.gt");
    Ov = B.CreateAnd(ANotZero, Over, "umulov.ov");
  } else if (II->getIntrinsicID() == Intrinsic::smul_with_overflow) {
    // ov = (a == -1 && b == INT_MIN) || (a != 0 && a != -1 && (a * b) / a != b)
    //
    // a == -1 has to be split out rather than folded into the division,
    // because sdiv INT_MIN, -1 is itself undefined.
    unsigned Bits = Ty->getIntegerBitWidth();
    Lo = B.CreateMul(A, Bv, "smulov.lo");
    Constant *Zero = ConstantInt::get(Ty, 0);
    Constant *One = ConstantInt::get(Ty, 1);
    Constant *NegOne = Constant::getAllOnesValue(Ty);
    Constant *IntMin = ConstantInt::get(Ty, APInt::getSignedMinValue(Bits));
    Value *ANotZero = B.CreateICmpNE(A, Zero, "smulov.anz");
    Value *AIsNegOne = B.CreateICmpEQ(A, NegOne, "smulov.aneg1");
    Value *NegOneOv = B.CreateAnd(AIsNegOne,
                                  B.CreateICmpEQ(Bv, IntMin, "smulov.bmin"),
                                  "smulov.ovneg1");
    Value *Divisible = B.CreateAnd(ANotZero, B.CreateNot(AIsNegOne),
                                   "smulov.divisible");
    Value *SafeA = B.CreateSelect(Divisible, A, One, "smulov.safea");
    Value *Back = B.CreateSDiv(Lo, SafeA, "smulov.back");
    Value *Ne = B.CreateICmpNE(Back, Bv, "smulov.ne");
    Ov = B.CreateOr(NegOneOv, B.CreateAnd(Divisible, Ne, "smulov.divov"),
                    "smulov.ov");
  } else {
    return false;
  }

  Value *Agg = UndefValue::get(II->getType());
  Agg = B.CreateInsertValue(Agg, Lo, 0, "ov.agg.lo");
  Agg = B.CreateInsertValue(Agg, Ov, 1, "ov.agg");
  II->replaceAllUsesWith(Agg);
  II->eraseFromParent();
  return true;
}

} // namespace

PreservedAnalyses HipLowerOverflowIntrinsicsPass::run(Module &M,
                                                      ModuleAnalysisManager &AM) {
  SmallVector<IntrinsicInst *, 8> Worklist;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *II = dyn_cast<IntrinsicInst>(&I))
          if (II->getIntrinsicID() == Intrinsic::umul_with_overflow ||
              II->getIntrinsicID() == Intrinsic::smul_with_overflow)
            Worklist.push_back(II);

  bool Changed = false;
  for (IntrinsicInst *II : Worklist)
    Changed |= expandCall(II);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
