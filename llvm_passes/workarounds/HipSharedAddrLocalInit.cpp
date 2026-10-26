//===- HipSharedAddrLocalInit.cpp -----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Initializes locals from __shared__ addresses in the function.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipSharedAddrLocalInit.h"

#include "LLVMSPIRV.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {
static bool refersToShared(const Value *V) {
  if (const auto *GV = dyn_cast<GlobalValue>(V))
    return GV->getAddressSpace() == SPIRV_WORKGROUP_AS;
  const auto *C = dyn_cast<Constant>(V);
  return C && any_of(C->operands(),
                     [](const Use &Op) { return refersToShared(Op.get()); });
}

// Stores C to Ptr, element by element where it refers to shared memory.
static void storeInit(Constant *C, AllocaInst *Ptr,
                      SmallVectorImpl<Value *> &Idx, IRBuilder<> &B) {
  if (refersToShared(C) &&
      (isa<ConstantArray>(C) || isa<ConstantStruct>(C))) {
    for (unsigned I = 0; I < C->getNumOperands(); ++I) {
      Idx.push_back(B.getInt32(I));
      storeInit(C->getAggregateElement(I), Ptr, Idx, B);
      Idx.pop_back();
    }
    return;
  }
  Type *Ty = Ptr->getAllocatedType();
  uint64_t Off =
      Ptr->getModule()->getDataLayout().getIndexedOffsetInType(Ty, Idx);
  B.CreateAlignedStore(C, B.CreateInBoundsGEP(Ty, Ptr, Idx),
                       commonAlignment(Ptr->getAlign(), Off));
}
} // namespace

PreservedAnalyses HipSharedAddrLocalInitPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  const DataLayout &DL = M.getDataLayout();
  SmallSetVector<GlobalVariable *, 4> Inits;
  for (Function &F : M)
    for (Instruction &I : make_early_inc_range(instructions(F))) {
      auto *Copy = dyn_cast<MemCpyInst>(&I);
      if (!Copy)
        continue;
      Value *Src = Copy->getRawSource();
      APInt Off(DL.getIndexTypeSizeInBits(Src->getType()), 0);
      auto *GV = dyn_cast<GlobalVariable>(
          Src->stripAndAccumulateConstantOffsets(DL, Off, true));
      if (!GV || !GV->isConstant() || !GV->hasDefinitiveInitializer() ||
          !refersToShared(GV->getInitializer()))
        continue;
      // Copy from a function-local instance of the initializer instead.
      IRBuilder<> B(&F.getEntryBlock(),
                    F.getEntryBlock().getFirstInsertionPt());
      AllocaInst *Init = B.CreateAlloca(GV->getValueType());
      B.SetInsertPoint(Copy);
      SmallVector<Value *, 4> Idx{B.getInt32(0)};
      storeInit(GV->getInitializer(), Init, Idx, B);
      Value *NewSrc = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), Init,
                                                   Off.getZExtValue());
      B.CreateMemCpy(Copy->getRawDest(), Copy->getDestAlign(), NewSrc,
                     commonAlignment(Init->getAlign(), Off.getZExtValue()),
                     Copy->getLength(), Copy->isVolatile());
      Copy->eraseFromParent();
      Inits.insert(GV);
    }
  for (GlobalVariable *GV : Inits) {
    GV->removeDeadConstantUsers();
    if (GV->use_empty() && GV->hasLocalLinkage())
      GV->eraseFromParent();
  }
  return Inits.empty() ? PreservedAnalyses::all() : PreservedAnalyses::none();
}
