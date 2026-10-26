//===- HipLowerHintIntrinsics.cpp -----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Erases, folds or lowers intrinsics llvm-spirv rejects, as LangRef permits.
//
// WORKAROUND(CHIP-SPV/chipStar#1633, KhronosGroup/SPIRV-LLVM-Translator#3990):
// llvm-spirv rejects these intrinsics. Remove each case once every supported
// SPIR-V producer accepts it.
//
// Copyright (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerHintIntrinsics.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

/// True if \p P provably points to addrspace(1), the only address space
/// OpenCL.std prefetch accepts.
static bool isGlobalPointer(Value *P) {
  P = getUnderlyingObject(P);
  // HIP kernel pointer arguments reach the body as inttoptr(ptrtoint(p1)).
  if (auto *I2P = dyn_cast<IntToPtrInst>(P))
    if (auto *P2I = dyn_cast<PtrToIntInst>(I2P->getOperand(0)))
      P = getUnderlyingObject(P2I->getPointerOperand());
  // __device__ variables reach it as inttoptr(load @__chip_var_addr_<name>).
  auto *I2P = dyn_cast<IntToPtrInst>(P);
  auto *Load = I2P ? dyn_cast<LoadInst>(I2P->getOperand(0)) : nullptr;
  bool DeviceVar = Load && isa<GlobalVariable>(Load->getPointerOperand());
  return (DeviceVar || isa<Argument, GlobalVariable>(P)) &&
         P->getType()->getPointerAddressSpace() == 1;
}

/// Rewrite one call. Returns true if \p II was replaced and erased.
static bool lowerCall(IntrinsicInst *II, const DataLayout &DL) {
  Type *Ty = II->getType();
  // Null means the call produces no value and is simply dropped.
  Value *Repl = nullptr;

  switch (II->getIntrinsicID()) {
  case Intrinsic::prefetch: {
    // WORKAROUND(CHIP-SPV/chipStar#1633, llvm/llvm-project#215505): that PR
    // prefetches any pointer, but OpenCL.std prefetch takes only a global one;
    // keep until kernel pointers reach producers global and others are dropped.
    Value *P = II->getArgOperand(0);
    if (!isGlobalPointer(P))
      break;
    IRBuilder<> B(II);
    FunctionCallee F = II->getModule()->getOrInsertFunction(
        "_Z8prefetchPU3AS1Kcm", B.getVoidTy(), B.getPtrTy(1), B.getInt64Ty());
    cast<Function>(F.getCallee())->setCallingConv(CallingConv::SPIR_FUNC);
    B.CreateCall(F, {B.CreateAddrSpaceCast(P, B.getPtrTy(1)), B.getInt64(1)})
        ->setCallingConv(CallingConv::SPIR_FUNC);
    break;
  }
  case Intrinsic::readcyclecounter:
  case Intrinsic::readsteadycounter:
    Repl = ConstantInt::get(Ty, 0);
    break;
  case Intrinsic::get_rounding:
    // The default OpenCL rounding mode is nearest even, FLT_ROUNDS encoding 1.
    Repl = ConstantInt::get(Ty, 1);
    break;
  case Intrinsic::allow_runtime_check:
    // Clang's default when no -lower-allow-check option is given.
    Repl = ConstantInt::getTrue(II->getContext());
    break;
  case Intrinsic::returnaddress:
  case Intrinsic::frameaddress:
    Repl = ConstantPointerNull::get(cast<PointerType>(Ty));
    break;
  case Intrinsic::objectsize:
    // No TLI: it only adds recognition of host libc allocators.
    Repl = lowerObjectSizeCall(II, DL, /*TLI=*/nullptr, /*MustSucceed=*/true);
    break;
  case Intrinsic::memcpy:
  case Intrinsic::memmove: {
    // WORKAROUND(CHIP-SPV/chipStar#1634, llvm/llvm-project#201904,
    // KhronosGroup/SPIRV-LLVM-Translator#3827): on LLVM 21 and 22 both
    // producers emit OpCopyMemorySized with the constant zero Size the spec
    // forbids. Remove once LLVM 21 and 22 are unsupported.
    auto *Len = dyn_cast<ConstantInt>(cast<MemTransferInst>(II)->getLength());
    if (!Len || !Len->isZero())
      return false;
    break;
  }
  case Intrinsic::memcpy_inline: {
    // OpCopyMemorySized must not have a constant zero Size.
    auto *MC = cast<MemCpyInst>(II);
    auto *Len = dyn_cast<ConstantInt>(MC->getLength());
    if (!Len || !Len->isZero())
      IRBuilder<>(MC).CreateMemCpy(MC->getRawDest(), MC->getDestAlign(),
                                   MC->getRawSource(), MC->getSourceAlign(),
                                   MC->getLength(), MC->isVolatile());
    break;
  }
  default:
    return false;
  }

  if (Repl)
    II->replaceAllUsesWith(Repl);
  II->eraseFromParent();
  return true;
}

} // namespace

PreservedAnalyses HipLowerHintIntrinsicsPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  SmallVector<IntrinsicInst *, 8> Worklist;
  SmallVector<IntrinsicInst *, 8> Copies;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *II = dyn_cast<IntrinsicInst>(&I))
        (isa<MemTransferInst>(II) ? Copies : Worklist).push_back(II);
  // A copy's length can be an intrinsic folded above, so visit copies last.
  Worklist.append(Copies.begin(), Copies.end());

  bool Changed = false;
  for (IntrinsicInst *II : Worklist)
    Changed |= lowerCall(II, M.getDataLayout());

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
