//===- HipLowerRoundIntrinsics.cpp ----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// libstdc++ declares std::llround and std::llrint for float as constexpr, which
// HIP turns into implicitly __host__ __device__ functions. They therefore win
// overload resolution in device code over anything chipStar could add (a plain
// __device__ overload with the same signature is rejected outright), and expand
// to __builtin_llroundf, i.e. an llvm.llround intrinsic. llvm-spirv has no
// translation for it:
//
//   InvalidFunctionCall: Unexpected llvm intrinsic: llvm.llround.i64.f32
//
// The failure surfaces at hipspv-link with no source location at all. Since the
// problem cannot be fixed in the headers, expand the intrinsic here: round in
// floating point with llvm.round / llvm.rint, which the translator maps to the
// OpenCL round and rint ExtInsts, then convert to the integer result. Calling
// the devicelib entry points instead does not work, because devicelib is linked
// in before the post-link passes run and only the symbols already referenced at
// that point are pulled in.
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

/// The intrinsic llvm.llround / llvm.llrint expands to, or not_intrinsic when
/// \p ID is something else.
static Intrinsic::ID getRoundingIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::llround:
    // llround rounds halfway cases away from zero, which is llvm.round.
    return Intrinsic::round;
  case Intrinsic::llrint:
    // llrint follows the current rounding mode, which is llvm.rint.
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
