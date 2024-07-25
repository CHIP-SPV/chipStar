//===- HipLowerZeroLengthArrays.cpp ---------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lowers occurrences of zero length array types which are not supported by
// the llvm-spirv.
//
// Copyright (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerZeroLengthArrays.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

#include <map>

#define PASS_NAME "hip-lower-zero-len-array-types"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

namespace {

/// Get legalized type tree
///
/// Return nullptr if the given type does not need legalization.
static Type* getLoweredTypeOrNull(Type *Ty) {
#if LLVM_VERSION_MAJOR < 16
  // Delve into the pointer element
  if (Ty->isPointerTy())
    if (auto *LoweredEltTy = getLoweredTypeOrNull(Ty->getPointerElementType()))
      return LoweredEltTy->getPointerTo(Ty->getPointerAddressSpace());
#endif

  // SPIRV-LLVM translator does not accept zero length arrays. Lower such
  // arrays to some arbitrary non-zero length.
  if (auto *ATy = dyn_cast<ArrayType>(Ty))
    if (ATy->getNumElements() == 0)
      return ArrayType::get(ATy->getElementType(), 1);

  return nullptr;
}

static bool hasUnsupportedType(Type *Ty) { return getLoweredTypeOrNull(Ty); }

static Constant *getLoweredConstantOrNull(Constant *C) {
  if (!C->getType()->isPointerTy())
    return nullptr;

  if (auto *GV = dyn_cast<GlobalVariable>(C)) {
    assert(!hasUnsupportedType(GV->getValueType()) &&
           "UNIMPLEMENTED: lower global var with a [0 x Ty] type.");
    return nullptr;
  }

  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    if (auto *GEP = dyn_cast<GEPOperator>(CE)) {
      auto *NewSrcTy = getLoweredTypeOrNull(GEP->getSourceElementType());
      auto *OrigPtr = cast<Constant>(GEP->getPointerOperand());
      auto *NewPtr = getLoweredConstantOrNull(OrigPtr);
      if (!NewSrcTy && !NewPtr)
        return nullptr;
      SmallVector<Value *> NewIndices;
      for (auto I = GEP->idx_begin(), E = GEP->idx_end(); I != E; I++)
        NewIndices.push_back(*I);

      auto *NewGEP = ConstantExpr::getGetElementPtr(
          (NewSrcTy ? NewSrcTy : GEP->getSourceElementType()),
          (NewPtr ? NewPtr : OrigPtr), NewIndices, GEP->isInBounds());
      return NewGEP;

    } else if (isa<AddrSpaceCastOperator>(CE) || isa<BitCastOperator>(CE) ||
               CE->getOpcode() == Instruction::IntToPtr) {
      auto *LoweredOpd0 = getLoweredConstantOrNull(CE->getOperand(0));
      auto *LoweredTy = getLoweredTypeOrNull(CE->getType());
      if (LoweredOpd0 || LoweredTy)
        return ConstantExpr::getCast(
            CE->getOpcode(), LoweredOpd0 ? LoweredOpd0 : CE->getOperand(0),
            LoweredTy ? LoweredTy : CE->getType());
      return nullptr;

    } else {
      dbgs() << "Unhandled constant expr: " << *CE << "\n";
      llvm_unreachable("");
    }
  }
  return nullptr;
}

static bool lowerZeroLengthArrayTypes(Function &F) {
  SmallVector<Instruction *> InstsToDelete;
  bool Modified = false;

  std::map<Value *, Value *> LoweredVals;
  auto recordLoweredValue = [&](Value *Original, Value *Lowered) -> void {
    assert(!LoweredVals.count(Original));
    LoweredVals[Original] = Lowered;
  };
  auto getLoweredValue = [&](Value *V) -> Value * {
    if (LoweredVals.count(V))
      return LoweredVals[V];
    // Perhaps the value didn't need lowering. Let's be sure about it.
    assert(!hasUnsupportedType(V->getType()));
    return V;
  };

  // First scan all constant pointer operands and lower them first.
  for (auto &BB : F)
    for (auto &I : BB)
      for (auto &U : I.operands()) {
        auto *Src = U.get();
        if (!isa<Constant>(Src))
          continue;
        auto *NewC = getLoweredConstantOrNull(cast<Constant>(Src));
        if (!NewC)
          continue;

        // If the original root constant does not have unsupported type we can
        // do RAUW right away. In the other case, the users (instructions) of
        // this constant also need lowering. For them we record the lowered
        // value. When the said users are lowered they'll request the
        // lowered value via getLoweredValue().
        auto *OrigTy = U.get()->getType();
        if (hasUnsupportedType(OrigTy))
          recordLoweredValue(Src, NewC);
        else
          Src->replaceAllUsesWith(ConstantExpr::getPointerCast(NewC, OrigTy));
        Modified |= true;
      }

  // Scan for instructions with references to unsupported type and lower them.
  for (auto &BB : F)
    for (auto &I : BB) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        auto *NewSrcTy = getLoweredTypeOrNull(GEP->getSourceElementType());
        if (!NewSrcTy)
          continue;
        auto *NewPtr = getLoweredValue(GEP->getPointerOperand());
        SmallVector<Value *> NewIndices;
        for (auto I = GEP->idx_begin(), E = GEP->idx_end(); I != E; I++)
          NewIndices.push_back(*I);
        GetElementPtrInst *NewGEP =
            GetElementPtrInst::Create(NewSrcTy, NewPtr, NewIndices, "", GEP);
        if (hasUnsupportedType(GEP->getType()))
          recordLoweredValue(GEP, NewGEP);
        else
          GEP->replaceAllUsesWith(NewGEP);
        InstsToDelete.push_back(GEP);
        Modified |= true;
      } else {
        // Catch instructions needing lowering.
        assert(!hasUnsupportedType(I.getType()));
      }
    }

  return Modified;
}

static bool lowerZeroLengthArrays(Module &M) {
  bool Modified = false;
  for (auto &F : M)
    Modified |= lowerZeroLengthArrayTypes(F);
  return Modified;
}

} // namespace

PreservedAnalyses HipLowerZeroLengthArraysPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {

  return lowerZeroLengthArrays(M) ? PreservedAnalyses::none()
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
                    FPM.addPass(HipLowerZeroLengthArraysPass());
                    return true;
                  }
                  return false;
                });
          }};
}
