//===- HipLowerVolatileAccesses.cpp ---------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A `volatile` access in HIP source means what it means in CUDA: clang lowers
// it to PTX ld.volatile / st.volatile, whose semantics the PTX ISA (8.4.2,
// "volatile Operation") defines as "equivalent to a relaxed memory operation
// with system-scope", a strong access that bypasses the core's L1 and observes
// other cores' relaxed atomics. HIP code written for that (Kokkos'
// volatile_load in the UnorderedMap insert list walk) polls memory another
// work-item publishes with __threadfence() followed by a relaxed CAS.
//
// SPIR-V has no such access. OpLoad with the Volatile memory operand only says
// the access "cannot be eliminated, duplicated, or combined with other
// accesses", and IGC serves it from L1 like any other load, so the poll reads
// stale data. The SPIR-V operation with the intended semantics is OpAtomicLoad
// / OpAtomicStore with Relaxed semantics at Device scope, which is what both
// SPIR-V producers emit for `load atomic ... syncscope("device") monotonic`.
//
// So every volatile load or store of a 32 or 64 bit integer, float or pointer
// through a global (addrspace 1) or generic (addrspace 4) pointer becomes the
// corresponding monotonic device-scope atomic. It stays volatile so nothing
// downstream merges or drops it. Floats and pointers go through the integer of
// the same width with a bitcast / ptrtoint / inttoptr around the access:
// OpAtomicLoad's result type must be an integer or float scalar, and the
// integer forms are the ones every OpenCL SPIR-V consumer implements (64 bit
// ones under Int64Atomics).
//
// Left as they are, and why:
//   - 8 and 16 bit values, which OpenCL SPIR-V consumers have no atomics for
//     (see HipLowerSubwordAtomics.cpp), and wider or vector values, which have
//     no atomic form at all.
//   - accesses whose pointer comes from a private (addrspace 0), constant
//     (addrspace 2) or work-group local (addrspace 3) object: private memory
//     is never shared, local memory never crosses a core, and atomics on the
//     Function storage class have no defined behaviour.
//   - under-aligned accesses: LLVM requires atomic loads and stores to be
//     naturally aligned. Clang never emits a volatile access that is not, so
//     these are reported and left volatile.
//   - accesses that are already atomic.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerVolatileAccesses.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include "PassPluginCompat.h"

#define PASS_NAME "hip-lower-volatile-accesses"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

namespace {

// SPIR-V address spaces as clang numbers them for spirv64.
constexpr unsigned PrivateAS = 0;
constexpr unsigned GlobalAS = 1;
constexpr unsigned ConstantAS = 2;
constexpr unsigned LocalAS = 3;
constexpr unsigned GenericAS = 4;

/// Whether the access may reach memory another core can also access. Kernel
/// argument pointers arrive as generic pointers (clang launders them through
/// ptrtoint / inttoptr, which InferAddressSpaces does not see through), so a
/// generic pointer is assumed to be global unless the object it is derived
/// from says otherwise.
bool mayBeShared(Value *Ptr) {
  unsigned AS = Ptr->getType()->getPointerAddressSpace();
  if (AS != GlobalAS && AS != GenericAS)
    return false;
  unsigned ObjAS =
      getUnderlyingObject(Ptr)->getType()->getPointerAddressSpace();
  return ObjAS != PrivateAS && ObjAS != ConstantAS && ObjAS != LocalAS;
}

/// The integer the access is performed on, or null when the value has no
/// 32 or 64 bit atomic form.
Type *atomicIntType(Type *ValTy, const DataLayout &DL) {
  if (!ValTy->isIntegerTy() && !ValTy->isFloatingPointTy() &&
      !ValTy->isPointerTy())
    return nullptr;
  uint64_t Bits = DL.getTypeStoreSizeInBits(ValTy);
  if (Bits != 32 && Bits != 64)
    return nullptr;
  if (ValTy->isIntegerTy() && ValTy->getIntegerBitWidth() != Bits)
    return nullptr; // i33 and friends round up to 64.
  return Type::getIntNTy(ValTy->getContext(), Bits);
}

bool isNaturallyAligned(Align A, Type *IntTy) {
  return A.value() * 8 >= IntTy->getIntegerBitWidth();
}

void lowerLoad(LoadInst *LI, Type *IntTy, SyncScope::ID SSID) {
  if (LI->getType() == IntTy) {
    LI->setAtomic(AtomicOrdering::Monotonic, SSID);
    return;
  }
  IRBuilder<> B(LI);
  LoadInst *Bits = B.CreateAlignedLoad(IntTy, LI->getPointerOperand(),
                                       LI->getAlign(), /*isVolatile=*/true,
                                       LI->getName() + ".bits");
  Bits->setAtomic(AtomicOrdering::Monotonic, SSID);
  Bits->setDebugLoc(LI->getDebugLoc());
  Value *V = LI->getType()->isPointerTy() ? B.CreateIntToPtr(Bits, LI->getType())
                                          : B.CreateBitCast(Bits, LI->getType());
  V->takeName(LI);
  LI->replaceAllUsesWith(V);
  LI->eraseFromParent();
}

void lowerStore(StoreInst *SI, Type *IntTy, SyncScope::ID SSID) {
  Value *V = SI->getValueOperand();
  if (V->getType() == IntTy) {
    SI->setAtomic(AtomicOrdering::Monotonic, SSID);
    return;
  }
  IRBuilder<> B(SI);
  Value *Bits = V->getType()->isPointerTy() ? B.CreatePtrToInt(V, IntTy)
                                            : B.CreateBitCast(V, IntTy);
  StoreInst *Raw = B.CreateAlignedStore(Bits, SI->getPointerOperand(),
                                        SI->getAlign(), /*isVolatile=*/true);
  Raw->setAtomic(AtomicOrdering::Monotonic, SSID);
  Raw->setDebugLoc(SI->getDebugLoc());
  SI->eraseFromParent();
}

bool lowerVolatileAccesses(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  SyncScope::ID DeviceSSID = F.getContext().getOrInsertSyncScopeID("device");
  SmallVector<std::pair<Instruction *, Type *>, 16> WorkList;
  for (auto &BB : F)
    for (auto &I : BB) {
      Value *Ptr = nullptr;
      Type *ValTy = nullptr;
      Align A;
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (!LI->isVolatile() || LI->isAtomic())
          continue;
        Ptr = LI->getPointerOperand();
        ValTy = LI->getType();
        A = LI->getAlign();
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (!SI->isVolatile() || SI->isAtomic())
          continue;
        Ptr = SI->getPointerOperand();
        ValTy = SI->getValueOperand()->getType();
        A = SI->getAlign();
      } else {
        continue;
      }
      Type *IntTy = atomicIntType(ValTy, DL);
      if (!IntTy || !mayBeShared(Ptr))
        continue;
      if (!isNaturallyAligned(A, IntTy)) {
        errs() << "warning: HipLowerVolatileAccesses: leaving under-aligned "
                  "volatile access alone in "
               << F.getName() << ": " << I << "\n";
        continue;
      }
      WorkList.emplace_back(&I, IntTy);
    }

  for (auto &[I, IntTy] : WorkList) {
    if (auto *LI = dyn_cast<LoadInst>(I))
      lowerLoad(LI, IntTy, DeviceSSID);
    else
      lowerStore(cast<StoreInst>(I), IntTy, DeviceSSID);
  }
  return !WorkList.empty();
}

} // namespace

PreservedAnalyses HipLowerVolatileAccessesPass::run(Function &F,
                                                    FunctionAnalysisManager &AM) {
  return lowerVolatileAccesses(F) ? PreservedAnalyses::none()
                                  : PreservedAnalyses::all();
}

#ifndef CHIP_COMBINED_PASS_PLUGIN
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_NAME, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_NAME) {
                    FPM.addPass(HipLowerVolatileAccessesPass());
                    return true;
                  }
                  return false;
                });
          }};
}
#endif // CHIP_COMBINED_PASS_PLUGIN
