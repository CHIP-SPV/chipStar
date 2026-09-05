//===- HipLowerPointerVectors.cpp -----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// WORKAROUND(CHIP-SPV/chipStar#1577, llvm/llvm-project): LLVM 23 forms
// <N x ptr> values that no SPIR-V path accepts. SROA folds a struct of
// pointers into a single vector-typed access, for example rocPRIM's
// scan_state in test/rocprim/test_device_reduce_by_key.cpp:
//
//   %v = load <3 x ptr addrspace(4)>, ptr %scan_state, align 8
//   store <3 x ptr addrspace(4)> %v, ptr %dst, align 32
//
// The in-tree SPIR-V backend aborts on an assertion in SPIRVEmitIntrinsics
// (FixedVectorType::get rejects a pointer element type), and llvm-spirv
// refuses the module unless SPV_INTEL_masked_gather_scatter is permitted.
// That extension is not a portable answer: IGC 2.11.29 accepts it but IGC
// 2.38.2 rejects it on bmg, dg2 and adl-s.
//
// A vector of integers, by contrast, is ordinary SPIR-V. This pass therefore
// carries such values as an integer vector of the same element width, so the
// pointer vector never exists by the time SPIR-V is emitted. Uses that need a
// real pointer get it back with inttoptr, which the backend handles.
//
// The rewrite covers loads and stores, which is the whole of the copy pattern
// SROA produces. A pointer vector reaching anything else is diagnosed rather
// than passed through, so the gap is visible at the pass instead of surfacing
// as an assertion inside SPIR-V emission.
//
// Remove this pass once the SPIR-V backend legalises <N x ptr> itself, or
// once LLVM stops forming the type for SPIR-V targets.
// See https://github.com/CHIP-SPV/chipStar/issues/1454.
//
// Copyright (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerPointerVectors.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/SmallVector.h"

#include <vector>

#define DEBUG_TYPE "hip-lower-pointer-vectors"

using namespace llvm;

namespace {

/// Returns the matching integer vector type for a vector of pointers, or
/// nullptr if \p Ty is not one. <3 x ptr addrspace(4)> becomes <3 x i64> on a
/// 64 bit target, so the value keeps its size and element count.
static FixedVectorType *getIntVecTypeFor(Type *Ty, const DataLayout &DL) {
  auto *VecTy = dyn_cast<FixedVectorType>(Ty);
  if (!VecTy || !VecTy->getElementType()->isPointerTy())
    return nullptr;
  unsigned AS = VecTy->getElementType()->getPointerAddressSpace();
  IntegerType *IntTy =
      IntegerType::get(Ty->getContext(), DL.getPointerSizeInBits(AS));
  return FixedVectorType::get(IntTy, VecTy->getNumElements());
}

/// Rewrites one load of a pointer vector into a load of an integer vector.
/// Users that consume the value as a vector of pointers are served with a
/// single inttoptr, so the pointer vector is confined to those users rather
/// than reaching the load itself.
static bool lowerLoad(LoadInst *LI, const DataLayout &DL) {
  FixedVectorType *IntVecTy = getIntVecTypeFor(LI->getType(), DL);
  if (!IntVecTy)
    return false;

  IRBuilder<> B(LI);
  auto *NewLI = B.CreateAlignedLoad(IntVecTy, LI->getPointerOperand(),
                                    LI->getAlign(), LI->isVolatile());
  NewLI->copyMetadata(*LI);

  // Only materialise a pointer vector again where something actually wants
  // one. In the copy pattern this pass targets, the sole user is a store,
  // which the store rewrite below turns into an integer store, leaving no
  // pointer vector at all.
  Value *AsPtrVec = B.CreateIntToPtr(NewLI, LI->getType());
  LI->replaceAllUsesWith(AsPtrVec);
  LI->eraseFromParent();
  return true;
}

/// Rewrites one store of a pointer vector into a store of an integer vector.
static bool lowerStore(StoreInst *SI, const DataLayout &DL) {
  Value *Val = SI->getValueOperand();
  FixedVectorType *IntVecTy = getIntVecTypeFor(Val->getType(), DL);
  if (!IntVecTy)
    return false;

  IRBuilder<> B(SI);
  // If the value came straight from a load this pass already rewrote, it is
  // an inttoptr of the integer vector we want. Use that operand directly
  // rather than adding a ptrtoint back: emitting the round trip would leave
  // <N x ptr> as the type of the intermediate values, which is exactly what
  // must not reach SPIR-V emission.
  Value *AsInts = nullptr;
  if (auto *ITP = dyn_cast<IntToPtrInst>(Val))
    if (ITP->getOperand(0)->getType() == IntVecTy)
      AsInts = ITP->getOperand(0);
  if (!AsInts)
    AsInts = B.CreatePtrToInt(Val, IntVecTy);
  auto *NewSI = B.CreateAlignedStore(AsInts, SI->getPointerOperand(),
                                     SI->getAlign(), SI->isVolatile());
  NewSI->copyMetadata(*SI);
  SI->eraseFromParent();
  return true;
}

} // namespace

PreservedAnalyses HipLowerPointerVectorsPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  const DataLayout &DL = M.getDataLayout();
  bool Changed = false;

  // Collect first: the rewrites erase instructions as they go.
  std::vector<LoadInst *> Loads;
  std::vector<StoreInst *> Stores;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I))
          Loads.push_back(LI);
        else if (auto *SI = dyn_cast<StoreInst>(&I))
          Stores.push_back(SI);
      }

  // Loads first, so that by the time a store is rewritten its value operand
  // is already the inttoptr this pass inserted and can be folded away.
  for (LoadInst *LI : Loads)
    Changed |= lowerLoad(LI, DL);
  for (StoreInst *SI : Stores)
    Changed |= lowerStore(SI, DL);

  // Drop inttoptr values left with no users once the stores folded them out.
  // Without this the pointer vector survives as the type of a dead value and
  // still reaches the backend.
  for (Function &F : M) {
    SmallVector<Instruction *, 8> Dead;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (isa<IntToPtrInst>(&I) && I.use_empty() &&
            I.getType()->isVectorTy())
          Dead.push_back(&I);
    for (Instruction *I : Dead) {
      I->eraseFromParent();
      Changed = true;
    }
  }

  // Only loads and stores are rewritten, which covers the copy pattern SROA
  // produces. A pointer vector reaching any other instruction (extractelement,
  // phi, select, a call argument) still cannot be emitted, so say so here
  // instead of letting SPIR-V emission abort on an assertion that names
  // neither this pass nor the issue it belongs to.
  auto reportSurvivor = [&](const Function &F, const Instruction &I) {
    std::string Msg;
    raw_string_ostream OS(Msg);
    OS << "HipLowerPointerVectors: a vector of pointers survives in '"
       << F.getName() << "', which SPIR-V cannot represent:\n  " << I
       << "\nOnly loads and stores of such vectors are lowered today. See "
          "https://github.com/CHIP-SPV/chipStar/issues/1577";
    report_fatal_error(StringRef(OS.str()), /*GenCrashDiag=*/false);
  };

  for (Function &F : M) {
    // An argument of this type has no defining instruction to point at, so it
    // is checked separately from the instruction walk below.
    for (Argument &A : F.args())
      if (getIntVecTypeFor(A.getType(), DL) && !F.empty())
        reportSurvivor(F, *F.getEntryBlock().begin());
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        if (getIntVecTypeFor(I.getType(), DL))
          reportSurvivor(F, I);
        for (const Use &U : I.operands())
          if (getIntVecTypeFor(U->getType(), DL))
            reportSurvivor(F, I);
      }
  }

  LLVM_DEBUG(dbgs() << "HipLowerPointerVectors: changed=" << Changed << "\n");
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
