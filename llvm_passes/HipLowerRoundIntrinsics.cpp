//===- HipLowerRoundIntrinsics.cpp ----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expands the type-crossing rounding intrinsics -- llvm.lround, llvm.llround,
// llvm.lrint and llvm.llrint -- which take a floating point value and return an
// integer. OpenCL.std has no instruction for that shape: every one of its
// rounding instructions (ceil, floor, rint, round, trunc) is float to float, so
// these four need a two instruction expansion rather than a 1:1 ExtInst. A
// producer that does not do that expansion itself rejects the module:
//
//   InvalidFunctionCall: Unexpected llvm intrinsic: llvm.llround.i64.f32
//
// and, with the in-tree SPIR-V backend:
//
//   LLVM ERROR: unable to legalize instruction: %8:iid(s64) = G_INTRINSIC_LRINT
//
// The failure surfaces at hipspv-link with no source location at all, and it
// cannot be headed off in the headers: libstdc++ declares these for float as
// constexpr, which HIP turns into implicitly __host__ __device__ functions, so
// they win overload resolution over anything chipStar could add (a plain
// __device__ overload with the same signature is rejected outright).
//
// So expand them here: round in floating point with llvm.round / llvm.rint,
// which both producers map to the OpenCL round and rint ExtInsts, then convert
// to the integer result. Calling the devicelib entry points instead does not
// work, because devicelib is linked in before the post-link passes run and only
// the symbols already referenced at that point are pulled in.
//
// All four are handled, not just the ones that happen to be untranslatable
// today, so that the set is defined by the shape of the intrinsic rather than
// by which producer version is in use. Of the four, only lround is handled by
// both producers as things stand, and both lower it to exactly this expansion
// (OpExtInst round, then OpConvertFToS), so covering it changes no emitted
// code. Taking the whole set also means the pass can be deleted in one go once
// the producers cover all of it.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerRoundIntrinsics.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include "PassPluginCompat.h"

#define PASS_NAME "hip-lower-round-intrinsics"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

/// The floating point rounding intrinsic a type-crossing rounder expands to, or
/// not_intrinsic when \p ID is something else.
static Intrinsic::ID getRoundingIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::lround:
  case Intrinsic::llround:
    // lround and llround round halfway cases away from zero, which is
    // llvm.round.
    return Intrinsic::round;
  case Intrinsic::lrint:
  case Intrinsic::llrint:
    // lrint and llrint follow the current rounding mode, which is llvm.rint.
    return Intrinsic::rint;
  default:
    return Intrinsic::not_intrinsic;
  }
}

static bool lowerRoundIntrinsics(Module &M) {
  SmallVector<CallInst *, 8> WorkList;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *Call = dyn_cast<CallInst>(&I))
          if (Function *Callee = Call->getCalledFunction())
            if (Call->arg_size() == 1 &&
                Call->getArgOperand(0)->getType()->isFloatingPointTy() &&
                getRoundingIntrinsic(Callee->getIntrinsicID()) !=
                    Intrinsic::not_intrinsic)
              WorkList.push_back(Call);

  for (auto *Call : WorkList) {
    Value *Arg = Call->getArgOperand(0);
    Intrinsic::ID Rounding =
        getRoundingIntrinsic(Call->getCalledFunction()->getIntrinsicID());

    IRBuilder<> Builder(Call);
    // Round in floating point, which llvm-spirv maps to the OpenCL round and
    // rint ExtInsts, then convert. The value is already integral at that point,
    // so the truncating conversion is exact.
    Value *Rounded = Builder.CreateUnaryIntrinsic(Rounding, Arg);
    Value *Result = Builder.CreateFPToSI(Rounded, Call->getType());

    Call->replaceAllUsesWith(Result);
    Call->eraseFromParent();
  }

  return !WorkList.empty();
}

PreservedAnalyses HipLowerRoundIntrinsicsPass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  return lowerRoundIntrinsics(M) ? PreservedAnalyses::none()
                                 : PreservedAnalyses::all();
}

#ifndef CHIP_COMBINED_PASS_PLUGIN
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_NAME, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_NAME) {
                    MPM.addPass(HipLowerRoundIntrinsicsPass());
                    return true;
                  }
                  return false;
                });
          }};
}
#endif // CHIP_COMBINED_PASS_PLUGIN
