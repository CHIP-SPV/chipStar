//===- HipCUDADV.cpp ------------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//


#include "HipCUDADV.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/IR/ReplaceConstant.h"

#include <iostream>
#include <set>

using namespace llvm;

class HipCUDADVImplPass : public ModulePass {

public:
  static char ID;
  HipCUDADVImplPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    return devirtImpl(M);
  }

  StringRef getPassName() const override {
    return "devirtualize virtual function calls ";
  }

  static bool devirtImpl(Module &M) {
    // Make devirtualizatoin 
    CUDADeVirt DVirt;
    // DVirt.apply(M); 

    return true;
  }
};

// Identifier variable for the pass
char HipCUDADVImplPass::ID = 0;
static RegisterPass<HipCUDADVImplPass>
    X("hip-cudadv",
      "evirtualize virtual function calls ");


// Pass hook for the new pass manager.
#if LLVM_VERSION_MAJOR > 11
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

PreservedAnalyses
HipCUDADVPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (HipCUDADVImplPass::devirtImpl(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-dyn-mem", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-dyn-mem") {
                    FPM.addPass(HipCUDADVPass());
                    return true;
                  }
                  return false;
                });
          }};
}

#endif // LLVM_VERSION_MAJOR > 11
