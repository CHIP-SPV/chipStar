//===- HipEmitLoweredNames.cpp --------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A pass to produce a file for mapping hiprtc name expressions to
// lowered/mangled function and global variable names needed to implement
// hiprtcGetLoweredName().
//
// This pass looks for a magic variable produced from:
//
//   extern "C" __device__ constexpr void *_chip_name_exprs[] = {
//     (void *)<name expression 1>,
//     (void *)<name expression 2>,
//     ...
//   };
//
// and writes the lowered names, respectively, into a file passed in another
// magic C-string variable '_chip_name_expr_output_file' (because pass plugin
// provided command-line options do not work yet).
//
// Finally, the magic variables are removed from the module.
//
// Copyright (c) 2021-22 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipEmitLoweredNames.h"

#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

#include <fstream>
#include <optional>

#define PASS_NAME "emit-lowered-names"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

namespace {

// Traverse constant expression to extract a C-string in it.
static std::optional<StringRef> getString(Constant *C) {
  C = C->stripPointerCasts();

  if (auto *GV = dyn_cast<GlobalVariable>(C)) {
    if (GV->hasInitializer())
      return getString(GV->getInitializer());
    return std::nullopt;
  }

  if (auto *CDS = dyn_cast<ConstantDataSequential>(C)) {
    if (CDS->isString())
      return CDS->getAsString();
    return std::nullopt;
  }

  return std::nullopt;
}

// Traverse constant expression to extract lowered name expressions.
static std::vector<std::string> getLoweredNames(Constant *C) {
  std::vector<std::string> Result;
  C = C->stripPointerCasts();

  if (auto *CA = dyn_cast<ConstantAggregate>(C)) {
    for (Value *Op : CA->operand_values()) {
      auto *COp = cast<Constant>(Op)->stripPointerCasts();
      if (auto *F = dyn_cast<Function>(COp))
        Result.emplace_back(F->hasName() ? F->getName() : "");
      else
        Result.emplace_back("");
    }
  }

  return Result;
}

static bool emitLoweredNames(Module &M) {
  auto *LoweredNamesGV = M.getGlobalVariable("_chip_name_exprs");
  if (!LoweredNamesGV)
    return false;
  assert(LoweredNamesGV->hasInitializer());

  std::ofstream OutputStream;
  auto *OutputGV = M.getGlobalVariable("_chip_name_expr_output_file");
  if (OutputGV) {
    assert(OutputGV->hasInitializer());
    if (auto OutputOpt = getString(OutputGV->getInitializer()))
      OutputStream = std::ofstream(OutputOpt->str());
  }

  if (OutputStream.is_open())
    for (const auto &LoweredName :
         getLoweredNames(LoweredNamesGV->getInitializer()))
      OutputStream << LoweredName << "\n";

  if (OutputGV && OutputGV->hasNUses(0))
    OutputGV->eraseFromParent();

  if (LoweredNamesGV->hasNUses(0))
    LoweredNamesGV->eraseFromParent();

  return true;
}

} // namespace

PreservedAnalyses HipEmitLoweredNamesPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {

  return emitLoweredNames(M) ? PreservedAnalyses::none()
                             : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_NAME, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_NAME) {
                    FPM.addPass(HipEmitLoweredNamesPass());
                    return true;
                  }
                  return false;
                });
          }};
}
