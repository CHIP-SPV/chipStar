//===- HipWarps.cpp -.-----------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR pass to handle kernels that are sensitive to warp width.
//
// (c) 2022-2023 Pekka Jääskeläinen / Intel
//===----------------------------------------------------------------------===//
//
// Pins the subgroup size of kernels that can depend on the warp width.
//===----------------------------------------------------------------------===//

#include "HipWarps.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Metadata.h>
#include "llvm/IR/Module.h"

#include "LLVMSPIRV.h"
#include "chipStarConfig.hh"

PreservedAnalyses HipWarpsPass::run(Module &Mod, ModuleAnalysisManager &AM) {

  // Functions that address __shared__ memory through a global variable.
  SmallPtrSet<Function *, 16> SharedUsers;
  for (GlobalVariable &GV : Mod.globals()) {
    if (GV.getAddressSpace() != SPIRV_WORKGROUP_AS)
      continue;
    SmallVector<User *, 16> Users(GV.users());
    while (!Users.empty()) {
      User *U = Users.pop_back_val();
      if (auto *I = dyn_cast<Instruction>(U))
        SharedUsers.insert(I->getFunction());
      else if (isa<ConstantExpr>(U) || isa<ConstantAggregate>(U))
        Users.append(U->user_begin(), U->user_end());
    }
  }

  // Pin kernels that reach a subgroup builtin or __shared__ memory, where warp
  // lock-step code exchanges data. Others stay free: IGC miscompiles some math
  // at SIMD32 on Xe-LP iGPUs.
  //
  // Kernels that reach an indirect call are never pinned: the driver then
  // passes zero for every argument after 'this' (TestIndirectCall.hip).
  struct Reach {
    bool Indirect = false;
    bool Lanes = false;
  };
  auto walk = [&](Function &Kernel) {
    Reach R;
    SmallPtrSet<Function *, 16> Seen{&Kernel};
    SmallVector<Function *, 16> Worklist{&Kernel};
    while (!Worklist.empty()) {
      Function *F = Worklist.pop_back_val();
      if (SharedUsers.count(F) || any_of(F->args(), [](Argument &A) {
            return A.getType()->isPointerTy() &&
                   A.getType()->getPointerAddressSpace() == SPIRV_WORKGROUP_AS;
          }))
        R.Lanes = true;
      for (Instruction &I : instructions(*F)) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (!CB || CB->isInlineAsm())
          continue;
        Function *Callee = CB->getCalledFunction();
        if (!Callee)
          R.Indirect = true; // A vtable slot, a callback, ...
        else if (Callee->isDeclaration())
          // __chip_ballot's body is linked at runtime (ballot_native.cl).
          R.Lanes |= Callee->getName().contains("sub_group") ||
                     Callee->getName().contains("__chip_ballot");
        else if (Seen.insert(Callee).second)
          Worklist.push_back(Callee);
      }
    }
    return R;
  };

  auto &Ctx = Mod.getContext();
  for (auto &F : Mod) {
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;
    Reach R = walk(F);
    if (R.Indirect || !R.Lanes)
      continue;

    IntegerType *I32Type = IntegerType::get(Ctx, 32);
    F.setMetadata("intel_reqd_sub_group_size",
                  MDNode::get(Ctx, ConstantAsMetadata::get(ConstantInt::get(
                                       I32Type, CHIP_DEFAULT_WARP_SIZE))));
  }

  // The metadata should not impact other chipStar passes.
  return PreservedAnalyses::all();
}
