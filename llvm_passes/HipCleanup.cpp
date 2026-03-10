//===- HipCleanup.cpp -----------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Removes chipStar/HIP internal globals that are not needed in the final
// SPIR-V output:
//   - __hip_cuid* and __hip_fatbin* (HIP compilation artifacts)
//   - __chipspv_device_heap (void* that clspv cannot handle)
//
// Functions that reference removed globals are stubbed (body replaced with
// ret zeroinitializer). Stubbing is transitive: non-kernel callers of stubbed
// functions are also stubbed.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipCleanup.h"

#include "../src/common.hh"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-cleanup"

using namespace llvm;

namespace {

static bool shouldRemoveGlobal(const GlobalVariable &GV) {
  if (!GV.hasName())
    return false;
  StringRef Name = GV.getName();

  // Keep __chip_var_* — those are lowered by HipGlobalVariablesPass
  if (Name.starts_with(ChipVarPrefix))
    return false;

  // Keep __chip_module_has_no_IGBAs — consumed by runtime for IGBA detection
  if (Name == "__chip_module_has_no_IGBAs")
    return false;

  // Remove HIP compilation artifacts
  if (Name.starts_with("__hip_cuid") || Name.starts_with("__hip_fatbin"))
    return true;

  // Remove __chipspv_device_heap — clspv cannot handle pointer-typed globals
  if (Name == "__chipspv_device_heap")
    return true;

  return false;
}

// Replace the function body with `ret zeroinitializer` (or `ret void`).
static void stubFunction(Function &F) {
  // Drop all references first to break use chains between basic blocks.
  F.dropAllReferences();

  // Delete all basic blocks.
  while (!F.empty())
    F.begin()->eraseFromParent();

  BasicBlock *BB = BasicBlock::Create(F.getContext(), "entry", &F);
  IRBuilder<> B(BB);

  Type *RetTy = F.getReturnType();
  if (RetTy->isVoidTy()) {
    B.CreateRetVoid();
  } else {
    B.CreateRet(Constant::getNullValue(RetTy));
  }
}

// Collect all non-kernel functions that directly use any of the removed
// globals.  Kernels are skipped — their individual uses are cleaned up in
// Step 5 so that other stores (e.g. device-variable resets) are preserved.
static void findDirectUsers(const SmallPtrSetImpl<GlobalVariable *> &Removed,
                            SmallPtrSetImpl<Function *> &ToStub) {
  for (auto *GV : Removed) {
    for (auto *U : GV->users()) {
      if (auto *I = dyn_cast<Instruction>(U)) {
        Function *F = I->getFunction();
        if (F->getCallingConv() != CallingConv::SPIR_KERNEL)
          ToStub.insert(F);
      } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        for (auto *CEU : CE->users()) {
          if (auto *I2 = dyn_cast<Instruction>(CEU)) {
            Function *F = I2->getFunction();
            if (F->getCallingConv() != CallingConv::SPIR_KERNEL)
              ToStub.insert(F);
          }
        }
      }
    }
  }
}

// Transitively stub non-kernel callers of already-stubbed functions.
static void transitivelyStubCallers(Module &M,
                                    SmallPtrSetImpl<Function *> &ToStub) {
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (auto &F : M) {
      if (ToStub.count(&F))
        continue;
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
        continue;
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *CB = dyn_cast<CallBase>(&I)) {
            if (auto *Callee = CB->getCalledFunction()) {
              if (ToStub.count(Callee)) {
                ToStub.insert(&F);
                Changed = true;
                goto next_func;
              }
            }
          }
        }
      }
    next_func:;
    }
  }
}

} // namespace

PreservedAnalyses HipCleanupPass::run(Module &M, ModuleAnalysisManager &AM) {
  bool ModuleChanged = false;

  // Step 1: Identify globals to remove.
  SmallPtrSet<GlobalVariable *, 16> ToRemove;
  for (auto &GV : M.globals()) {
    if (shouldRemoveGlobal(GV))
      ToRemove.insert(&GV);
  }

  // Step 2: Find functions that reference removed globals.
  SmallPtrSet<Function *, 16> ToStub;
  findDirectUsers(ToRemove, ToStub);

  // Step 3: Transitively stub non-kernel callers.
  transitivelyStubCallers(M, ToStub);

  // Step 4: Stub the functions.
  for (auto *F : ToStub) {
    LLVM_DEBUG(dbgs() << "HipCleanup: stubbing function " << F->getName()
                      << "\n");
    stubFunction(*F);
    ModuleChanged = true;
  }

  // Step 5: Remove the globals.
  // First, delete all instructions that use the globals to avoid leaving
  // poison values in kernels like __chip_reset_non_symbols.
  for (auto *GV : ToRemove) {
    LLVM_DEBUG(dbgs() << "HipCleanup: removing global " << GV->getName()
                      << "\n");
    SmallVector<Instruction *, 8> UsersToDelete;
    for (auto *U : GV->users()) {
      if (auto *I = dyn_cast<Instruction>(U))
        UsersToDelete.push_back(I);
      else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        for (auto *CEU : CE->users())
          if (auto *I2 = dyn_cast<Instruction>(CEU))
            UsersToDelete.push_back(I2);
      }
    }
    for (auto *I : UsersToDelete)
      I->eraseFromParent();
    GV->removeDeadConstantUsers();
    if (GV->use_empty())
      GV->eraseFromParent();
    else {
      GV->replaceAllUsesWith(PoisonValue::get(GV->getType()));
      GV->eraseFromParent();
    }
    ModuleChanged = true;
  }

  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
