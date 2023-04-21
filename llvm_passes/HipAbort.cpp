//===- HipAbort.cpp -.-----------------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR pass to handle kernels with abort() calls.
//
// (c) 2023 CHIP-SPV developers
//     2022 Pekka Jääskeläinen / Parmance for Argonne National Laboratory
//===----------------------------------------------------------------------===//
//
// HIP supports an abort() that is specified as
// "...will terminate the process execution from inside the kernel."
// https://rocmdocs.amd.com/en/latest/Programming_Guides/Kernel_language.html
//
// We approximate this behavior with a global host-readable variable that is set
// when abort() is called. The runtime, after detecting the variable is set
// calls the process abort() immediately. This LLVM pass treats abort() calls
// so we do not execute any kernel code after calling them.
//
// This is handled by converting all function calls (that potentially call abort
// or _are_ abort, perhaps recursively) as following:
//
// func_call();
//
// ..is converted to...
//
// func_call(); if (__chipspv_abort_called) return; [poison?]
//
// Thus it forces the call tree be unwound similar to an exception in C++.
//
// Another option would have been to just busy loop in abort() after setting the
// flag and poll at host side for the abort flag. However, it then makes the
// device busy and reading the global variable might not work (since it
// currently requires kernel execution).
//
// Note: This doesn't handle all cases. Especially kernels with barrier calls
// likely won't work if there are diverging abort calls as some of the threads
// return early and might never reach the barrier due to a race condition.
// A proper all-cases covering implementation would be best done with
// SPIR-V/OpenCL support, which affects how barriers are treated.
//===----------------------------------------------------------------------===//

#include "HipAbort.h"

#include "../src/common.hh"

#include <set>
#include <iostream>

__attribute__((optnone)) PreservedAnalyses
HipAbortPass::run(Module &Mod, ModuleAnalysisManager &AM) {

  // The abort calls are made to undefined abort decl thus should not get
  // inlined.
  GlobalValue *AbortF = Mod.getNamedValue("__chipspv_abort");

  if (AbortF == nullptr) {
    // Mark modules that do not call abort by just the global flag variable.
    // Ugly, but should allow avoiding the kernel call to check the global
    // variable.
    GlobalVariable *AbortFlag = Mod.getGlobalVariable(ChipDeviceAbortFlagName);
    if (AbortFlag != nullptr) {
      AbortFlag->replaceAllUsesWith(
          Constant::getNullValue(AbortFlag->getType()));
      AbortFlag->eraseFromParent();
    }
    return PreservedAnalyses::all();
  }

  assert(AbortF->isDeclaration());

  auto &Ctx = Mod.getContext();
  auto *Int32Ty = IntegerType::get(Ctx, 32);
  auto *Int32OneConst = ConstantInt::get(Int32Ty, 1);

  GlobalVariable *AbortFlag = Mod.getGlobalVariable(ChipDeviceAbortFlagName);
  assert(AbortFlag != nullptr);

  bool Modified = false;
  std::set<CallInst *> CallsToHandle;
  std::set<CallInst *> AbortCallsToHandle;
  for (auto &F : Mod) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (I.isDebugOrPseudoInst() || !isa<CallInst>(I))
          continue;
        CallInst *Call = cast<CallInst>(&I);
        CallsToHandle.insert(Call);
        if (Call->getCalledFunction()->getName() == "__chipspv_abort")
          AbortCallsToHandle.insert(Call);
      }
    }
  }

  // Use a single abort exit BB per function.
  std::map<Function *, BasicBlock *> AbortReturnBlocks;

  for (auto Call : CallsToHandle) {
    Instruction *InstrAfterCall = Call->getNextNonDebugInstruction();
    Function *Func = Call->getParent()->getParent();
    llvm::BasicBlock *OrigBB = InstrAfterCall->getParent();
    llvm::BasicBlock *FallThrough = OrigBB->splitBasicBlock(InstrAfterCall);
    llvm::BasicBlock *ReturnBlock = AbortReturnBlocks[Func];
    if (ReturnBlock == nullptr) {
      ReturnBlock = AbortReturnBlocks[Func] =
          BasicBlock::Create(Ctx, "abort_return_bb", Func);

      Type *RetTy = Func->getFunctionType()->getReturnType();
      ReturnInst::Create(Ctx,
                         RetTy->isVoidTy() ? nullptr : UndefValue::get(RetTy),
                         ReturnBlock);
    }

    InstrAfterCall = Call->getNextNonDebugInstruction();
    IRBuilder<> B(InstrAfterCall);
    LoadInst *FlagLoad =
        B.CreateLoad(AbortFlag->getValueType(), AbortFlag, true, "abort_flag");
    Value *FlagCmp = B.CreateICmpNE(
        FlagLoad, ConstantInt::get(AbortFlag->getValueType(), 0));

    // Replace the fallthrough terminator BR in the old BB.
    B.CreateCondBr(FlagCmp, ReturnBlock, FallThrough);

    OrigBB->getTerminator()->eraseFromParent();
  }

  for (auto Call : AbortCallsToHandle) {
    // Convert the calls to the abort() decl to stores to the abort flag.
    IRBuilder<> B(Call);
    B.CreateStore(Int32OneConst, AbortFlag, true);
    Call->eraseFromParent();
  }

  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
