//===- HipOffsetBeforeCast.cpp --------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Rewrites a GEP over an addrspacecast of a global as a cast of the GEP.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipOffsetBeforeCast.h"

#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"

using namespace llvm;

namespace {
// WORKAROUND(CHIP-SPV/chipStar#1703, no upstream report): the SPIR-V backend
// indexes an initializer's byte offset into the global's own type. Remove
// when it offsets by bytes.
static bool retypeToBytes(GlobalVariable &GV) {
  Type *I8 = Type::getInt8Ty(GV.getContext());
  if (auto *Ty = dyn_cast<ArrayType>(GV.getValueType());
      Ty && Ty->getElementType() == I8)
    return true;
  if (!GV.hasLocalLinkage() || !GV.hasDefinitiveInitializer())
    return false;
  const DataLayout &DL = GV.getParent()->getDataLayout();
  Constant *Init = GV.getInitializer();
  auto *Ty = ArrayType::get(I8, DL.getTypeAllocSize(GV.getValueType()));
  Constant *Bytes = ConstantAggregateZero::get(Ty);
  if (!Init->isNullValue()) {
    // The most constituents one OpConstantComposite can hold.
    if (Ty->getNumElements() > 65532)
      return false;
    SmallVector<uint8_t, 64> Data;
    for (uint64_t I = 0; I < Ty->getNumElements(); ++I) {
      // Null for a byte of a pointer or of undef.
      auto *B = dyn_cast_or_null<ConstantInt>(
          ConstantFoldLoadFromConst(Init, I8, APInt(64, I), DL));
      if (!B)
        return false;
      Data.push_back(B->getZExtValue());
    }
    Bytes = ConstantDataArray::get(GV.getContext(), Data);
  }
  if (!GV.getAlign())
    GV.setAlignment(DL.getPreferredAlign(&GV));
  GV.replaceInitializer(Bytes);
  return true;
}
} // namespace

PreservedAnalyses HipOffsetBeforeCastPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  // Literal indices: rewriting one entry then cannot change, and free,
  // another.
  auto IsOffset = [](User *U, User *CE) {
    auto *GEP = dyn_cast<GEPOperator>(U);
    return GEP && isa<ConstantExpr>(U) && GEP->getPointerOperand() == CE &&
           all_of(GEP->indices(),
                  [](Value *Idx) { return isa<ConstantInt>(Idx); });
  };
  auto IsCast = [](User *U) {
    auto *CE = dyn_cast<ConstantExpr>(U);
    return CE && CE->getOpcode() == Instruction::AddrSpaceCast;
  };
  SmallVector<std::pair<GlobalVariable *, GEPOperator *>, 8> Work;
  for (GlobalVariable &GV : M.globals()) {
    GV.removeDeadConstantUsers();
    // Retype GV only if a rewrite below reaches a constant.
    if (none_of(GV.users(), [&](User *CE) {
          return IsCast(CE) && any_of(CE->users(), [&](User *U) {
                   return IsOffset(U, CE) &&
                          !all_of(U->users(), IsaPred<Instruction>);
                 });
        }) ||
        !retypeToBytes(GV))
      continue;
    for (User *CE : GV.users())
      if (IsCast(CE))
        for (User *U : CE->users())
          if (IsOffset(U, CE))
            Work.emplace_back(&GV, cast<GEPOperator>(U));
  }
  for (auto [GV, GEP] : Work) {
    SmallVector<Value *, 4> Idx(GEP->idx_begin(), GEP->idx_end());
    Constant *Offset = ConstantExpr::getGetElementPtr(
        GEP->getSourceElementType(), GV, Idx, GEP->getNoWrapFlags(),
        GEP->getInRange());
    // Direct instruction operands compile fine with the original shape.
    GEP->replaceUsesWithIf(
        ConstantExpr::getAddrSpaceCast(Offset, GEP->getType()),
        [](Use &U) { return !isa<Instruction>(U.getUser()); });
  }
  return Work.empty() ? PreservedAnalyses::all() : PreservedAnalyses::none();
}
