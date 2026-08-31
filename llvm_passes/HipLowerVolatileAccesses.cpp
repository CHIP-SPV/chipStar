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
// with system-scope", an access that bypasses the core's private cache. HIP
// code written for that (Kokkos' volatile_load in the UnorderedMap insert list
// walk) polls memory another work-item publishes with a plain store,
// __threadfence() and a relaxed CAS.
//
// SPIR-V's Volatile memory operand carries none of that. It only says the
// access "cannot be eliminated, duplicated, or combined with other accesses",
// so IGC serves it from the Xe core's L1 like any other load and the poll reads
// stale data (chipStar issue #1508).
//
// The operand that does say something about caching is Nontemporal, on the very
// same OpLoad / OpStore. It is a hint, so a consumer that ignores it is exactly
// as correct as it is today, and on IGC it is the one cache control that beats
// every other rule: LSCCacheHints::SetupLscCacheCtrl
// (IGC/Compiler/CISACodeGen/LSCCacheHintsPass.cpp) maps an access carrying
// !nontemporal to LSC_L1UC_L3UC before it looks at anything else, i.e. the
// access is served past the core's L1.
//
// So every volatile load and store through a global (addrspace 1) or generic
// (addrspace 4) pointer is marked !nontemporal, which both SPIR-V producers
// emit as the Nontemporal memory operand. The access stays a plain, volatile
// load or store of its original type.
//
// It deliberately does NOT become an atomic, which is what an earlier version
// of this pass did (`load atomic ... syncscope("device") monotonic`, emitted as
// OpAtomicLoad / OpAtomicStore). Two facts make that unsound:
//
//   - IGC has no atomic load or store message. It implements OpAtomicLoad as
//     `atomic_or(p, 0)` and OpAtomicStore as an atomic exchange
//     (IGC/BiFModule/Implementation/atomics.cl), so both are read-modify-write
//     transactions against the memory.
//   - Level Zero lets a device report atomics as unsupported per allocation
//     kind (ze_memory_access_cap_flags_t, ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC).
//     A PVC on Aurora reports hostAllocCapabilities = RW with no ATOMIC, so a
//     GPU atomic on hipHostMalloc / hipMallocManaged memory faults with
//     "AtomicAccessViolation ... banned: 1" and the process aborts.
//
// A volatile pointer carries no provenance that tells host memory from device
// memory, so the lowering cannot pick per access whether an atomic is legal and
// must stay non-atomic.
//
// Left alone, and why:
//   - accesses whose pointer comes from a private (addrspace 0), constant
//     (addrspace 2) or work-group local (addrspace 3) object: private memory is
//     never shared, work-group local memory does not leave the core, and
//     constant memory does not change while a kernel runs.
//   - accesses that are already atomic: those bypass L1 by themselves and carry
//     an ordering of their own.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerVolatileAccesses.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
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

bool lowerVolatileAccesses(Function &F) {
  SmallVector<Instruction *, 16> WorkList;
  for (auto &BB : F)
    for (auto &I : BB) {
      Value *Ptr = nullptr;
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (!LI->isVolatile() || LI->isAtomic())
          continue;
        Ptr = LI->getPointerOperand();
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (!SI->isVolatile() || SI->isAtomic())
          continue;
        Ptr = SI->getPointerOperand();
      } else {
        continue;
      }
      if (!mayBeShared(Ptr))
        continue;
      if (I.getMetadata(LLVMContext::MD_nontemporal))
        continue;
      WorkList.push_back(&I);
    }

  if (WorkList.empty())
    return false;

  LLVMContext &Ctx = F.getContext();
  // The node LLVM's LangRef defines for !nontemporal, and the one the SPIR-V
  // translator reads to set the Nontemporal memory operand: a single i32 1.
  MDNode *Nontemporal = MDNode::get(
      Ctx, ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1)));
  for (Instruction *I : WorkList)
    I->setMetadata(LLVMContext::MD_nontemporal, Nontemporal);
  return true;
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
