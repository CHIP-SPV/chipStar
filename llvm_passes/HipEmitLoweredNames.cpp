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
// This pass looks for magic variables analogous to 1):
//
//   extern "C" __device__ constexpr auto *_chip_name_exprs_<num>
//     = <name expression>;
//
// and writes the lowered names into a file passed in another magic C-string
// variable '_chip_name_expr_output_file' (because pass plugin provided
// command-line options do not work yet). The entries in the file are ordered by
// <num>.
//
// Finally, the all the mentioned magic variables are removed from the module.
//
// Copyright (c) 2021-22 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipEmitLoweredNames.h"

#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#if LLVM_VERSION_MAJOR >= 22
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif
#include "llvm/Support/CommandLine.h"

#include <fstream>
#include <optional>
#include <map>

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

// Traverse global variables to extract lowered name expressions.
static std::vector<std::string>
getLoweredNames(std::map<unsigned, GlobalVariable *> NameExprGVs) {
  std::vector<std::string> Result;
  for (auto &[Ignored, GV] : NameExprGVs) {
    auto *C = GV->getInitializer()->stripPointerCasts();
    if (auto *F = dyn_cast<Function>(C))
      Result.emplace_back(F->hasName() ? F->getName() : "");
    else
      Result.emplace_back("");
  }

  return Result;
}

static bool emitLoweredNames(Module &M) {

  std::map<unsigned, GlobalVariable *> NameExprGVs;
  for (auto &GV : M.globals()) {
    if (!GV.hasName())
      continue;
    StringRef GVName = GV.getName();
    if (GVName.starts_with("_chip_name_expr_output_file") ||
        !GVName.consume_front("_chip_name_expr_"))
      continue;

    assert(GV.hasInitializer() && "_chip_name_expr_<num> is uninitialized!");

    unsigned Num;
    [[maybe_unused]] bool Error = GVName.consumeInteger(/*radix=*/10, Num);
    assert(!Error && "Unexpected _chip_name_expr name format!");

    assert(NameExprGVs.count(Num) == 0 &&
           "Duplicate _chip_name_expr_<num> variables!");
    NameExprGVs[Num] = &GV;
  }

  if (NameExprGVs.empty())
    return false;

  std::ofstream OutputStream;
  auto *OutputGV = M.getGlobalVariable("_chip_name_expr_output_file");
  if (OutputGV) {
    assert(OutputGV->hasInitializer());
    if (auto OutputOpt = getString(OutputGV->getInitializer()))
      OutputStream = std::ofstream(OutputOpt->str());
  }

  if (OutputStream.is_open())
    for (const auto &LoweredName : getLoweredNames(NameExprGVs))
      OutputStream << LoweredName << "\n";

  if (OutputGV && OutputGV->hasNUses(0))
    OutputGV->eraseFromParent();

  for (auto &[Ignored, GV] : NameExprGVs)
    if (GV->hasNUses(0))
      GV->eraseFromParent();

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
