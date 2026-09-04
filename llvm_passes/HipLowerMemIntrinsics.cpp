//===- HipLowerMemIntrinsics.cpp ------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expand llvm.memset intrinsics to loops before handing the bitcode to
// llvm-spirv translator. The translator expands memset intrinsics to an
// emulation function with a name encoding the destination pointer type
// (e.g. "spirv.spirv.llvm_memset_p0_i32..."). During reverse translation,
// llvm-spirv attempts to recreate the original memset intrinsic which can be
// incorrect if we are jumping from opaque pointer world to typed pointer world
// or vice versa. LLVM module verification may fail with the following message:
//
//   Fails to verify module: Intrinsic name not mangled correctly for
//   type arguments!
//
// This issue has been filed here:
// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/2128
//
// llvm.memcpy does not have this issue as it gets translated to actual SPIR-V
// instruction (OpCopyMemorySized).
//
// llvm.memcpy.inline is expanded here too, for a different reason: no SPIR-V
// producer accepts it. The translator rejects it with "InvalidFunctionCall:
// Unexpected llvm intrinsic: llvm.memcpy.inline.p4.p4.i64", because its writer
// has a case for Intrinsic::memcpy but none for Intrinsic::memcpy_inline, and
// the in-tree SPIR-V backend fails to legalize G_MEMCPY_INLINE for the generic
// address space clang emits for HIP. The memset form needs no separate handling
// because MemSetInst::classof already covers Intrinsic::memset_inline, while
// MemCpyInst::classof covers plain llvm.memcpy as well, which must keep its
// direct OpCopyMemorySized translation and so is filtered out below.
//
// A loop keeps what LangRef guarantees for the intrinsic: "The behavior of
// 'llvm.memcpy.inline.*' is equivalent to the behavior of 'llvm.memcpy.*', but
// the generated code is guaranteed not to call any external functions."
//
// (c) 2023 Henry Linjamäki / Intel
//===----------------------------------------------------------------------===//
#include "HipLowerMemIntrinsics.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Passes/PassBuilder.h>
#include "PassPluginCompat.h"
#include <llvm/Transforms/Utils/LowerMemIntrinsics.h>

using namespace llvm;

static bool lowerMemIntrinsics(Function &F, FunctionAnalysisManager &AM) {
  SmallPtrSet<MemSetInst *, 8> MemSets;
  SmallPtrSet<MemCpyInst *, 8> InlineMemCpys;
  for (auto &BB : F)
    for (auto &I : BB) {
      if (auto *MemSet = dyn_cast<MemSetInst>(&I))
        MemSets.insert(MemSet);
      else if (auto *MemCpy = dyn_cast<MemCpyInst>(&I))
        if (MemCpy->getIntrinsicID() == Intrinsic::memcpy_inline)
          InlineMemCpys.insert(MemCpy);
    }

  for (auto *MemSet : MemSets) {
    expandMemSetAsLoop(MemSet);
    MemSet->eraseFromParent();
  }

  if (!InlineMemCpys.empty()) {
    const auto &TTI = AM.getResult<TargetIRAnalysis>(F);
    for (auto *MemCpy : InlineMemCpys) {
      expandMemCpyAsLoop(MemCpy, TTI);
      MemCpy->eraseFromParent();
    }
  }

  return !MemSets.empty() || !InlineMemCpys.empty();
}

PreservedAnalyses HipLowerMemIntrinsicsPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  return lowerMemIntrinsics(F, AM) ? PreservedAnalyses::none()
                                   : PreservedAnalyses::all();
}

#ifndef CHIP_COMBINED_PASS_PLUGIN
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-lower-mem-intrinsics",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-lower-mem-intrinsics") {
                    FPM.addPass(HipLowerMemIntrinsicsPass());
                    return true;
                  }
                  return false;
                });
          }};
}
#endif // CHIP_COMBINED_PASS_PLUGIN
