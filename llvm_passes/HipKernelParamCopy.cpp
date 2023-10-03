//===- HipKernelParamCopy.cpp ---------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A pass to copy struct kernel params.
// largely follows the nvptx implementation: https://llvm.org/doxygen/NVPTXLowerArgs_8cpp_source.html
// For kernel function, check if the pointer types args have byVal attribute set
// if yes, check if they are any store instructions on the ptrs
// if yes, make a local copy and replace all the uses of args with the pointer to local copy
//===----------------------------------------------------------------------===//

#include "HipKernelParamCopy.h"

#include "LLVMSPIRV.h"
#include "../src/common.hh"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <map>
#include <string>
#include <iostream>

using namespace std;

#define PASS_ID "hip-kernel-param-copy"
#define DEBUG_TYPE PASS_ID

using namespace llvm;

namespace {

void handleByValParam(Argument *Arg){
  Function *Func = Arg->getParent();
  Instruction *FirstInst = &(Func->getEntryBlock().front());
  Type *StructType = Arg->getParamByValType();
  assert(StructType && "Missing byval type");

  auto IsALoadChain = [&](Value *Start) {
    SmallVector<Value *, 16> ValuesToCheck = {Start};
    auto IsALoadChainInstr = [](Value *V) -> bool {
      return (isa<GetElementPtrInst>(V) || isa<BitCastInst>(V) || isa<LoadInst>(V));
    };

    while (!ValuesToCheck.empty()) {
      Value *V = ValuesToCheck.pop_back_val();
      if (!IsALoadChainInstr(V)) {
       LLVM_DEBUG(dbgs() << "Need a copy of " << *Arg << " because of " << *V<< "\n");
        (void)Arg;
        return false;
      }
      if (!isa<LoadInst>(V))
	llvm::append_range(ValuesToCheck, V->users());
    }
    return true;
  };

  if ( !llvm::all_of(Arg->users(), IsALoadChain)) {

    //create a temporary copy
    const DataLayout &DL = Func->getParent()->getDataLayout();
    unsigned AS = DL.getAllocaAddrSpace();
    AllocaInst *AllocA = new AllocaInst(StructType, AS, Arg->getName(), FirstInst);
    AllocA->setAlignment(Func->getParamAlign(Arg->getArgNo())
                            .value_or(DL.getPrefTypeAlign(StructType)));
    Arg->replaceAllUsesWith(AllocA);
    LoadInst *LI = new LoadInst(StructType, Arg, Arg->getName(),
                  false, AllocA->getAlign(), FirstInst);
    new StoreInst(LI, AllocA, FirstInst);
  }
}

static bool handleKernelParamCopy(Module &M) {
  for (auto &F : M) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      for (Argument &Arg : F.args()) {
        if (Arg.getType()->isPointerTy()) {
          if (Arg.hasByValAttr())
	    handleByValParam(&Arg);
        }
      }
    }
  }

  return true;
}

} // namespace

PreservedAnalyses HipKernelParamCopyPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  return handleKernelParamCopy(M) ? PreservedAnalyses::none()
    : PreservedAnalyses::all();
}

namespace {
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_ID, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
              [](StringRef Name, ModulePassManager &MPM,
                  ArrayRef<PassBuilder::PipelineElement>) {
                if (Name == PASS_ID) {
                  MPM.addPass(HipKernelParamCopyPass());
		  return true;
                }
                return false;

	     }
            );
          }
        };
}
} // namespace


