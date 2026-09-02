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
// stale data (chipStar issue #1508). Measured on a PVC and a B570, an unmarked
// volatile reader misses a writer work-group's store 35840 times out of 35840.
//
// So every volatile load and store through a global (addrspace 1) or generic
// (addrspace 4) pointer becomes a relaxed device-scope atomic, which both
// SPIR-V producers emit as OpAtomicLoad / OpAtomicStore with Relaxed semantics
// at Device scope. The access keeps its type and its volatility; only the
// ordering and syncscope are added. The explicit "device" syncscope matters:
// the translator at LLVM 17 hardcodes Device for atomic loads, while newer ones
// map the default scope to CrossDevice.
//
// Why an atomic and not the Nontemporal memory operand, which an earlier
// version of this pass used: Nontemporal is only a hint ("Hints that the
// accessed address is not likely to be accessed again in the near future"),
// which is not even true of a poll loop, and IGC maps it to LSC_L1UC_L3UC,
// uncached at BOTH levels. That combination is actively wrong here. On an Arc
// A380 the marking made every reader observe the publish flag and then read a
// stale payload, 35840 out of 35840, deterministically, because the writer's
// stores are write-back while the reader's marked loads bypass L3 as well as
// L1. Measured with ocloc on the module this pass produced:
//
//   writer:  store.ugm.d32x4t.a64.wb.wb   IGC widened the loop stores and
//            store.ugm.d32x2t.a64.wb.wb   dropped the cache control entirely
//   reader:  load.ugm.d32x1t.a64.uc.uc    bypasses L1 AND L3, reads memory
//
// So the hint was both too strong (giving up L3) and not durable (silently
// discarded when IGC merges stores). An atomic is neither: it cannot be widened
// away and it is a requirement rather than an advisory operand.
//
// The cost is that an atomic is only legal where the allocation supports one.
// Level Zero lets a device report atomics as unsupported per allocation kind
// (ze_memory_access_cap_flags_t, ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC), and a PVC
// on Aurora reports hostAllocCapabilities = RW with no ATOMIC. Measured there,
// one operation per allocation kind per process:
//
//   kind      load    store   rmw     plain
//   device    OK      OK      OK      OK
//   pinned    OK      BAN     BAN     OK
//   managed   OK      BAN     BAN     OK
//
// where BAN is "AtomicAccessViolation ... banned: 1" and an abort. hipMallocManaged
// was fixed by CHIP-SPV/chipStar#1514, which backs it with single-device shared
// USM whose sharedSingleDeviceAllocCapabilities do report ATOMIC. hipHostMalloc
// still uses zeMemAllocHost and therefore still aborts on a volatile STORE on
// PVC; that is CHIP-SPV/chipStar#1489 and it gates this pass on Aurora.
// hipHostRegister has no route at all, since PVC reports
// sharedSystemAllocCapabilities = 0x00.
//

// Left alone, and why:
//   - accesses every one of whose underlying objects is a private (addrspace
//     0), constant (addrspace 2) or work-group local (addrspace 3) object:
//     private memory is never shared, work-group local memory does not leave
//     the core, and constant memory does not change while a kernel runs. A
//     pointer a phi or a select joins is left alone only when every branch of
//     the join is one of those, so a kernel choosing between __shared__ scratch
//     and a pointer argument is still marked.
//   - accesses that are already atomic: those bypass L1 by themselves and carry
//     an ordering of their own.
//   - accesses that are not naturally aligned 32 or 64 bit scalars: vectors,
//     aggregates and the narrow widths gain nothing from the hint, and a
//     OpenCL SPIR-V allows atomics on 32 bit types only, 64 bit under a
//     capability, so narrower or wider shapes have no legal atomic form.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerVolatileAccesses.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
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
/// generic pointer is assumed to be global unless every object it can be
/// derived from says otherwise.
bool mayBeShared(Value *Ptr) {
  unsigned AS = Ptr->getType()->getPointerAddressSpace();
  if (AS != GlobalAS && AS != GenericAS)
    return false;
  // The plural form is what reaches the objects behind a pointer that a phi or
  // a select joins: the singular getUnderlyingObject stops at such a join and
  // hands back the join itself, whose address space is the generic one both
  // branches were cast to, which would report every branch as shared even when
  // none of them is. That shape is ordinary HIP: a kernel choosing between
  // __shared__ scratch and one of its pointer arguments produces it.
  SmallVector<const Value *, 4> Objects;
  getUnderlyingObjects(Ptr, Objects);
  // No object identified: the pointer may be anything, so assume shared.
  if (Objects.empty())
    return true;
  // Shared unless every object the access can reach is core-private.
  return any_of(Objects, [](const Value *Obj) {
    unsigned ObjAS = Obj->getType()->getPointerAddressSpace();
    return ObjAS != PrivateAS && ObjAS != ConstantAS && ObjAS != LocalAS;
  });
}

/// Whether the rewrite is legal on this access's type. It is restricted to
/// naturally aligned 32 and 64 bit scalars, which is the flag poll issue #1508
/// is about, and which is also the only width an atomic may have here: the
/// OpenCL SPIR-V environment spec allows atomic instructions on 32 bit types
/// only, with 64 bit under a capability, so an 8 or 16 bit atomic would be
/// invalid SPIR-V (see CHIP-SPV/chipStar#1497 and #1553).
///
/// Wider coverage is not available: the OpenCL SPIR-V environment spec allows
/// atomic instructions on 32 bit types only, with 64 bit under a capability, so
/// an 8 or 16 bit atomic would simply be invalid SPIR-V. Vectors and aggregates
/// have no atomic form at all. Note this restriction is no longer the
/// CHIP-SPV/chipStar#1551 workaround it was under the !nontemporal marking:
/// that was about LLVM's x86 back end aborting on a narrow non-temporal load,
/// which no longer applies now that no metadata is attached.
bool isMarkableType(Type *Ty, Align Alignment, const DataLayout &DL) {
  // Scalars only: this is false for vectors and aggregates.
  if (!Ty->isIntegerTy() && !Ty->isFloatingPointTy() && !Ty->isPointerTy())
    return false;
  TypeSize Size = DL.getTypeStoreSize(Ty);
  if (Size.isScalable())
    return false;
  uint64_t Bytes = Size.getFixedValue();
  if (Bytes != 4 && Bytes != 8)
    return false;
  return Alignment.value() >= Bytes;
}

bool lowerVolatileAccesses(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  SmallVector<Instruction *, 16> WorkList;
  for (auto &BB : F)
    for (auto &I : BB) {
      Value *Ptr = nullptr;
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (!LI->isVolatile() || LI->isAtomic())
          continue;
        if (!isMarkableType(LI->getType(), LI->getAlign(), DL))
          continue;
        Ptr = LI->getPointerOperand();
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (!SI->isVolatile() || SI->isAtomic())
          continue;
        if (!isMarkableType(SI->getValueOperand()->getType(), SI->getAlign(),
                            DL))
          continue;
        Ptr = SI->getPointerOperand();
      } else {
        continue;
      }
      if (!mayBeShared(Ptr))
        continue;
      // Already atomic accesses are filtered above; nothing else to skip.
      WorkList.push_back(&I);
    }

  if (WorkList.empty())
    return false;

  LLVMContext &Ctx = F.getContext();
  // Relaxed ordering at device scope: the access must observe other agents'
  // stores, and nothing more is claimed. The explicit "device" syncscope
  // matters, as the translator at LLVM 17 hardcodes Device for atomic loads
  // while newer ones map the default scope to CrossDevice.
  // Which lowering is baked in is a build-time decision: the cache-control
  // decorations make the module declare OpCapability CacheControlsINTEL, and a
  // consumer that does not support it rejects the whole module rather than
  // ignoring the capability (rusticl: clBuildProgram -11, "spirv_to_nir
  // failed"). Since one SPIR-V module has to load on whatever device the
  // runtime picks, the choice cannot be deferred. Build with
  // -DCHIP_ATOMICS_CACHE_BYPASS_WORKAROUND=ON for such targets.
  //
  // CHIP_VOLATILE_LOWERING=atomic|cachectl overrides it at compile time, for
  // A/B testing one build against both lowerings.
  bool UseAtomics =
#ifdef CHIP_ATOMICS_CACHE_BYPASS_WORKAROUND
      true;
#else
      false;
#endif
  if (const char *Env = getenv("CHIP_VOLATILE_LOWERING"))
    UseAtomics = StringRef(Env) == "atomic";

  if (!UseAtomics) {
    // SPV_INTEL_cache_controls: UncachedINTEL at cache level 0, the level
    // closest to the processing unit. IGC maps it to an L1-uncached but
    // L3-cached access (.uc.ca on pvc, dg2 and bmg), which is what a volatile
    // access needs and is what AMD's own volatile lowering does with glc/dlc.
    Type *I32 = Type::getInt32Ty(Ctx);
    auto MakeDeco = [&](unsigned DecoId) {
      Metadata *Ops[] = {ConstantAsMetadata::get(ConstantInt::get(I32, DecoId)),
                         ConstantAsMetadata::get(ConstantInt::get(I32, 0)),
                         ConstantAsMetadata::get(ConstantInt::get(I32, 0))};
      return MDNode::get(Ctx, {MDNode::get(Ctx, Ops)});
    };
    for (Instruction *I : WorkList) {
      bool IsLoad = isa<LoadInst>(I);
      Value *Ptr = IsLoad ? cast<LoadInst>(I)->getPointerOperand()
                          : cast<StoreInst>(I)->getPointerOperand();
      Type *AccTy = IsLoad ? I->getType()
                           : cast<StoreInst>(I)->getValueOperand()->getType();
      // The decoration attaches to the pointer instruction, not the access:
      // the translator asserts if it is put on the load or store itself. The
      // index must be non-constant-foldable in spirit, but a fresh GEP is kept
      // by both producers because it carries metadata.
      auto *G = GetElementPtrInst::CreateInBounds(
          AccTy, Ptr, {ConstantInt::get(Type::getInt64Ty(Ctx), 0)}, "vptr",
          I->getIterator());
      G->setMetadata("spirv.Decorations",
                     MakeDeco(IsLoad ? 6442 /* CacheControlLoadINTEL */
                                     : 6443 /* CacheControlStoreINTEL */));
      I->setOperand(IsLoad ? 0 : 1, G);
    }
    return true;
  }

  SyncScope::ID DeviceScope = Ctx.getOrInsertSyncScopeID("device");
  for (Instruction *I : WorkList) {
    if (auto *LI = dyn_cast<LoadInst>(I)) {
      Type *Ty = LI->getType();
      if (!Ty->isPointerTy()) {
        LI->setAtomic(AtomicOrdering::Monotonic, DeviceScope);
        continue;
      }
      // An atomic may not have a pointer result type: the OpenCL SPIR-V
      // environment requires OpAtomicLoad's Result Type to be an integer or
      // float scalar, and spirv-val rejects the pointer form. Load the
      // same-width integer instead and convert back, which leaves every user
      // of the original value untouched.
      IRBuilder<> B(LI);
      Type *IntTy = B.getIntNTy(DL.getTypeStoreSizeInBits(Ty).getFixedValue());
      LoadInst *NewLI = B.CreateAlignedLoad(IntTy, LI->getPointerOperand(),
                                            LI->getAlign(), LI->isVolatile());
      NewLI->setAtomic(AtomicOrdering::Monotonic, DeviceScope);
      Value *AsPtr = B.CreateIntToPtr(NewLI, Ty);
      LI->replaceAllUsesWith(AsPtr);
      LI->eraseFromParent();
    } else {
      auto *SI = cast<StoreInst>(I);
      Value *V = SI->getValueOperand();
      if (!V->getType()->isPointerTy()) {
        SI->setAtomic(AtomicOrdering::Monotonic, DeviceScope);
        continue;
      }
      IRBuilder<> B(SI);
      Type *IntTy = B.getIntNTy(
          DL.getTypeStoreSizeInBits(V->getType()).getFixedValue());
      StoreInst *NewSI = B.CreateAlignedStore(B.CreatePtrToInt(V, IntTy),
                                              SI->getPointerOperand(),
                                              SI->getAlign(), SI->isVolatile());
      NewSI->setAtomic(AtomicOrdering::Monotonic, DeviceScope);
      SI->eraseFromParent();
    }
  }
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
