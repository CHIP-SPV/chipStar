//===- HipIGBADetector.cpp --------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass analyzes module to find out if it has potential indirect global
// buffer accesses (IGBA). The outcome of the analysis is stored in a magic
// variable for the chipStar runtime:
//
//   uint8_t __chip_module_has_no_IGBAs = <result>;
//
// Where the result is one if the are no potential IGBAs and otherwise it is
// zero.
//
// If there would be an IGPA in the module, there has to to be a load
// instruction with a pointer operand which is either loaded from memory or
// crafted from an integer (which OTOH is loaded from somewhere else). The
// analysis is very simple and naive: we look for pointer load and inttoptr
// instructions in the whole module. If we see any, we conclude there are
// potential IGBAs. Downsides of this are that
//
// * may-have-IGBAs is concluded even tough only one kernel has IGBAs an
//   others don't
//
// * unoptimized modules (-O0) will likely likely result in may-have-IGBAs
// * conclusion
//
// The motivation for this analysis is to reduce clSetKernelExecInfo() calls in
// the OpenCL backend.
//
// (c) 2024 Henry Linjamäki / Intel
//===----------------------------------------------------------------------===//
#include "HipIGBADetector.h"

#include "LLVMSPIRV.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include "llvm/IR/Module.h"
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#define PASS_NAME "hip-igpa-detector"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

// Iterates through all instructions in the module to find potential
// Indirect Global Buffer Accesses (IGBAs).
// An IGBA is considered potential if:
//  1. An IntToPtrInst is found where the source operand is a true integer
//     (not resulting from a PtrToIntInst, which often indicates an
//     address-space cast or similar benign transformation).
//  2. A LoadInst is found that loads a pointer type from memory.
static bool hasPotentialIGBAs(Module &M) {
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *ITP = dyn_cast<IntToPtrInst>(&I)) {
          Value *Op = ITP->getOperand(0);
          // Skip benign ptr→int→ptr sequences lowered from address-space casts
          // or other pointer manipulations where an integer representation is
          // temporarily used.
          if (isa<PtrToIntInst>(Op))
            continue;
          // Only treat as IGBA if the source is a true integer, not a pointer
          // type that was cast to integer and then back to pointer (which is
          // covered by the above check typically, but this is an additional guard).
          if (!Op->getType()->isIntegerTy())
            continue;
          {
            LLVM_DEBUG(dbgs() << "Found genuine IntToPtrInst in function "
                              << F.getName() << ": " << ITP << "\n");
            return true;
          }
          // Old pointer-type check is now redundant and removed.
        }
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          // If an instruction loads a pointer from memory, it's a potential IGBA.
          if (LI->getType()->isPointerTy()) {
            LLVM_DEBUG(dbgs() << "Found pointer LoadInst in function " << F.getName()
                              << ": " << *LI << "\n");
            return true;
          }
        }
      }
    }
  }
  return false;
}

// Detects potential IGBAs in the module and sets a magic global variable
// "__chip_module_has_no_IGBAs" to indicate the result.
// '1' means no potential IGBAs were found.
// '0' means potential IGBAs were found.
// Returns true if the module was modified (i.e., the global variable was added),
// false otherwise (e.g., if the variable already existed).
static bool detectIGBAs(Module &M) {
  constexpr auto *MagicVarName = "__chip_module_has_no_IGBAs";

  if (M.getGlobalVariable(MagicVarName))
    return false; // Bail out: the module has already been processed.

  bool Result = hasPotentialIGBAs(M);
  LLVM_DEBUG(dbgs() << "Has IGBAs: " << Result << "\n");

  // The magic variable stores the *opposite* of the detection result:
  // If Result is true (IGBAs found), store 0.
  // If Result is false (no IGBAs found), store 1.
  auto *Init = ConstantInt::get(IntegerType::get(M.getContext(), 8), !Result);
  (void)new GlobalVariable(
      M, Init->getType(), true,
      // Mark the GV as external for keeping it alive at least until the
      // chipStar runtime reads it.
      GlobalValue::ExternalLinkage, Init, MagicVarName, nullptr,
      GlobalValue::NotThreadLocal /* Default value */,
      // Global-scope variables may not have Function storage class.
      // TODO: use private storage class?
      SPIRV_CROSSWORKGROUP_AS);

  return true;
}

PreservedAnalyses HipIGBADetectorPass::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  return detectIGBAs(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_NAME, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_NAME) {
                    MPM.addPass(HipIGBADetectorPass());
                    return true;
                  }
                  return false;
                });
          }};
}
