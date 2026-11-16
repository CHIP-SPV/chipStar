//===- HipFunctionPointerAS.cpp -------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Move function pointers stored in global variable initializers into the
// generic address space.
//
// WORKAROUND(CHIP-SPV/chipStar#1373, llvm/llvm-project#212452): clang types
// every C++ vtable component with the default globals address space and
// addrspacecasts function addresses into it, which SPIR-V cannot express.
// Remove this pass when https://github.com/llvm/llvm-project/pull/212452
// (vtable components emitted in the generic address space on SPIR-V targets)
// is in every supported LLVM.
//
// Clang emits C++ vtables for SPIR-V targets with every slot in the global
// address space, so each virtual function slot is
//
//   addrspacecast (ptr @_ZN7DerivedD2Ev to ptr addrspace(1))
//
// SPIR-V has no cast between two named storage classes: OpPtrCastToGeneric
// goes from Function/Workgroup/CrossWorkgroup into Generic and
// OpGenericCastToPtr goes back out, but a Function (plain 'ptr') to
// CrossWorkgroup cast has no instruction, so the translator rejects the module
// with
//
//   InvalidModule: Invalid SPIR-V module: Casts from private/local/global
//   address space are allowed only to generic
//
// This pass retypes such globals so their slots hold generic (address space 4)
// pointers, then repairs the slot loads and the indirect call sites that read
// them. Generic is the storage class SPV_INTEL_function_pointers overloads
// onto CodeSectionINTEL, and OpPtrCastToGeneric accepts the Function-class
// pointer both SPIR-V producers emit for a function's address, so every cast
// left in the module has a direction SPIR-V defines and Intel's consumers
// accept the result. See CHIP-SPV/chipStar#1373.
//
// The trigger is semantic, not the Itanium '_ZTV' name: any global whose
// initializer casts a Function into a non-generic address space, which also
// covers hand written dispatch tables. Only the element type of the table
// changes; the vptr stored inside the object is an ordinary data pointer and
// stays in the global address space. Non-function slots (offset-to-top, RTTI)
// move to generic as well, which is a cast SPIR-V defines.
//
// Validator note: with opaque pointers the SPIR-V producers scavenge the slot
// pointee as i8, so the slot constants come out as OpSpecConstantOp
// PtrCastToGeneric from a pointer to OpTypeFunction to a generic pointer to
// i8. That does not satisfy the core rule that both sides of the cast point to
// the same type; SPIRV-Tools from 2026-03 checks it, IGC and NEO accept it.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipFunctionPointerAS.h"

#include "LLVMSPIRV.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-function-pointer-as"

using namespace llvm;

namespace {

constexpr unsigned GenericAS = SPIRV_GENERIC_AS;

/// True for 'addrspacecast (ptr @SomeFunction to ptr addrspace(N))' with N
/// not the generic address space: the construct the SPIR-V translator rejects.
static bool isFunctionCastToNonGeneric(const Constant *C) {
  const auto *CE = dyn_cast<ConstantExpr>(C);
  return CE && CE->getOpcode() == Instruction::AddrSpaceCast &&
         isa<Function>(CE->getOperand(0)) &&
         CE->getType()->getPointerAddressSpace() != GenericAS;
}

/// Rewrite pointer-typed constant \p C so its type becomes 'ptr addrspace(4)'.
static Constant *coercePtrToGeneric(Constant *C) {
  auto *PtrTy = cast<PointerType>(C->getType());
  if (PtrTy->getAddressSpace() == GenericAS)
    return C;

  auto *NewTy = PointerType::get(C->getContext(), GenericAS);
  if (isa<ConstantPointerNull>(C))
    return ConstantPointerNull::get(NewTy);
  if (isa<PoisonValue>(C))
    return PoisonValue::get(NewTy);
  if (isa<UndefValue>(C))
    return UndefValue::get(NewTy);

  // Rebuild an existing addrspacecast from its source rather than stacking a
  // second cast on top of an illegal one.
  if (auto *CE = dyn_cast<ConstantExpr>(C))
    if (CE->getOpcode() == Instruction::AddrSpaceCast)
      return ConstantExpr::getAddrSpaceCast(CE->getOperand(0), NewTy);

  // Plain data such as the RTTI slot: global -> generic is OpPtrCastToGeneric.
  return ConstantExpr::getAddrSpaceCast(C, NewTy);
}

/// Rebuild \p C so that every function pointer it holds is generic. Returns
/// \p C itself when nothing had to change. With \p Force every pointer in \p C
/// moves to generic, which keeps an array homogeneous once one of its
/// elements had to move: the null and RTTI slots of a vtable follow the
/// function slots.
static Constant *rewriteInitializer(Constant *C, bool Force = false) {
  if (isFunctionCastToNonGeneric(C) || (Force && C->getType()->isPointerTy()))
    return coercePtrToGeneric(C);

  auto *CA = dyn_cast<ConstantAggregate>(C);
  if (!CA)
    return C;

  SmallVector<Constant *, 8> Elts;
  bool Changed = false;
  for (Use &Op : CA->operands()) {
    Constant *New = rewriteInitializer(cast<Constant>(Op), Force);
    Changed |= New != Op;
    Elts.push_back(New);
  }
  if (!Changed)
    return C;

  if (auto *CS = dyn_cast<ConstantStruct>(CA))
    // Vtables are literal structs; a literal struct with the rewritten member
    // types is what a retyped one has to be.
    return ConstantStruct::getAnon(Elts, CS->getType()->isPacked());

  if (!Force)
    for (unsigned I = 0, E = Elts.size(); I != E; ++I)
      Elts[I] = rewriteInitializer(CA->getOperand(I), /*Force=*/true);
  if (isa<ConstantVector>(CA))
    return ConstantVector::get(Elts);
  return ConstantArray::get(ArrayType::get(Elts.front()->getType(), Elts.size()),
                            Elts);
}

/// Replace \p Old with a global holding \p NewInit. Opaque pointers make the
/// address of the global the same LLVM type regardless of the value type, so
/// users can simply be redirected.
static void retypeGlobal(Module &M, GlobalVariable *Old, Constant *NewInit) {
  auto *New = new GlobalVariable(
      M, NewInit->getType(), Old->isConstant(), Old->getLinkage(), NewInit, "",
      nullptr, Old->getThreadLocalMode(), Old->getAddressSpace(),
      Old->isExternallyInitialized());
  New->copyAttributesFrom(Old);
  New->setComdat(Old->getComdat());
  New->copyMetadata(Old, 0);

  // A constant getelementptr records its source element type, so
  // replaceAllUsesWith would leave GEPs that still claim the old value type
  // while pointing at the retyped global. Classes with virtual bases hit this:
  // the VTT holds one 'getelementptr inrange (...) ({...}, ptr @_ZTC...)' per
  // construction vtable, and the translator this pass was written against
  // aborted in transConstantUse() on the disagreement; current producers
  // tolerate it. Rebuild those GEPs against the new value type first.
  SmallVector<ConstantExpr *, 8> GEPUsers;
  for (User *U : Old->users())
    if (auto *CE = dyn_cast<ConstantExpr>(U))
      if (CE->getOpcode() == Instruction::GetElementPtr &&
          cast<GEPOperator>(CE)->getSourceElementType() == Old->getValueType())
        GEPUsers.push_back(CE);
  for (ConstantExpr *CE : GEPUsers) {
    auto *GEP = cast<GEPOperator>(CE);
    SmallVector<Value *, 4> Idxs(CE->op_begin() + 1, CE->op_end());
    CE->replaceAllUsesWith(ConstantExpr::getGetElementPtr(
        New->getValueType(), New, Idxs, GEP->getNoWrapFlags(),
        GEP->getInRange()));
  }

  Old->replaceAllUsesWith(New);
  New->takeName(Old);
  Old->eraseFromParent();
}

/// Produce a value equivalent to \p V but typed 'ptr addrspace(4)'.
///
/// The interesting case is a load of a vtable slot. Its result type has to
/// change: after the table was retyped the memory holds a generic pointer, and
/// a load claiming to read a global pointer would translate to an OpLoad whose
/// result type disagrees with the pointee. Casting after the load would not
/// help for the same reason. Phis and selects of slots are rebuilt in generic
/// with their operands rewritten in turn. Anything else gets an addrspacecast,
/// which SPIR-V defines from every named storage class.
///
/// \p Generic caches the replacement of every value already rewritten, so a
/// slot shared by several calls is retyped once and a phi cycle closes on the
/// rebuilt phi. \p Dead collects the instructions left without a purpose.
static Value *makeCalleeGeneric(Value *V, Instruction *InsertBefore,
                                DenseMap<Value *, Value *> &Generic,
                                SmallSetVector<Instruction *, 8> &Dead) {
  auto *PtrTy = dyn_cast<PointerType>(V->getType());
  if (!PtrTy || PtrTy->getAddressSpace() == GenericAS)
    return V;
  auto *GenericPtrTy = PointerType::get(V->getContext(), GenericAS);

  if (auto *C = dyn_cast<Constant>(V))
    return coercePtrToGeneric(C);

  // A cast back into a named storage class: its source is already what we
  // want. Readers of a slot load that was retyped for another call see one.
  if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V))
    if (ASC->getSrcAddressSpace() == GenericAS) {
      Dead.insert(ASC);
      return ASC->getPointerOperand();
    }

  if (auto It = Generic.find(V); It != Generic.end())
    return It->second;

  if (auto *LI = dyn_cast<LoadInst>(V)) {
    Value *Ptr = LI->getPointerOperand();

    // The slot address is normally a getelementptr whose source element type
    // is the old slot type. The byte offset is the same either way, but a
    // typed-pointer consumer wants the GEP and the load to agree.
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr))
      if (auto *SrcPtrTy = dyn_cast<PointerType>(GEP->getSourceElementType()))
        if (SrcPtrTy->getAddressSpace() != GenericAS) {
          SmallVector<Value *, 4> Idx(GEP->idx_begin(), GEP->idx_end());
          auto *NewGEP = GetElementPtrInst::Create(
              GenericPtrTy, GEP->getPointerOperand(), Idx, GEP->getName(),
              GEP->getIterator());
          NewGEP->setNoWrapFlags(GEP->getNoWrapFlags());
          NewGEP->setDebugLoc(GEP->getDebugLoc());
          GEP->replaceAllUsesWith(NewGEP);
          Dead.insert(GEP);
          Ptr = NewGEP;
        }

    IRBuilder<> B(LI);
    auto *NewLI = B.CreateAlignedLoad(GenericPtrTy, Ptr, LI->getAlign(),
                                      LI->isVolatile(), LI->getName());
    NewLI->setOrdering(LI->getOrdering());
    NewLI->setSyncScopeID(LI->getSyncScopeID());
    NewLI->copyMetadata(*LI);
    Generic[LI] = NewLI;

    // Other readers of the old load still want the original address space;
    // generic -> global is OpGenericCastToPtr.
    if (!LI->use_empty()) {
      auto *Back = cast<Instruction>(B.CreateAddrSpaceCast(NewLI, PtrTy));
      LI->replaceAllUsesWith(Back);
      Dead.insert(Back);
    }
    Dead.insert(LI);
    return NewLI;
  }

  if (isa<PHINode>(V) || isa<SelectInst>(V)) {
    auto *I = cast<Instruction>(V);
    Instruction *NewI = I->clone();
    NewI->mutateType(GenericPtrTy);
    NewI->insertBefore(I->getIterator());
    Generic[I] = NewI;
    for (Use &U : NewI->operands()) {
      if (U->getType() != PtrTy)
        continue; // The select condition.
      Instruction *IP = NewI;
      if (auto *Phi = dyn_cast<PHINode>(NewI))
        IP = Phi->getIncomingBlock(U)->getTerminator();
      U.set(makeCalleeGeneric(U.get(), IP, Generic, Dead));
    }
    Dead.insert(I);
    return NewI;
  }

  return IRBuilder<>(InsertBefore).CreateAddrSpaceCast(V, GenericPtrTy);
}

/// Retype every global whose initializer holds a function pointer cast into a
/// non-generic address space. Returns true if the module changed.
static bool fixGlobalInitializers(Module &M) {
  bool Changed = false;
  for (GlobalVariable &GV : llvm::make_early_inc_range(M.globals())) {
    if (!GV.hasInitializer())
      continue;
    Constant *NewInit = rewriteInitializer(GV.getInitializer());
    if (NewInit == GV.getInitializer())
      continue;
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": retyping " << GV.getName() << "\n");
    retypeGlobal(M, &GV, NewInit);
    Changed = true;
  }
  return Changed;
}

/// Repair calls whose callee is not in the generic address space.
static bool fixIndirectCalls(Module &M) {
  SmallVector<CallBase *, 8> Calls;
  for (Function &F : M)
    for (Instruction &I : instructions(F)) {
      auto *CB = dyn_cast<CallBase>(&I);
      if (!CB || isa<Function>(CB->getCalledOperand()))
        continue;
      unsigned AS = CB->getCalledOperand()->getType()->getPointerAddressSpace();
      // The program address space is what a plain function pointer uses and
      // translates fine; generic is what this pass produces.
      if (AS != GenericAS && AS != M.getDataLayout().getProgramAddressSpace())
        Calls.push_back(CB);
    }
  if (Calls.empty())
    return false;

  DenseMap<Value *, Value *> Generic;
  SmallSetVector<Instruction *, 8> Dead;
  for (CallBase *CB : Calls) {
    Value *Callee = CB->getCalledOperand();
    // A constant cast of a function is a direct call in disguise; strip it.
    if (auto *F = dyn_cast<Function>(Callee->stripPointerCasts()))
      CB->setCalledOperand(F);
    else
      CB->setCalledOperand(makeCalleeGeneric(Callee, CB, Generic, Dead));
  }
  // Drop the originals: whatever lost its last user, iterating since erasing
  // one frees its operands, and a rebuilt phi that only fed itself.
  SmallVector<Instruction *, 8> Pending(Dead.begin(), Dead.end());
  for (bool Again = true; Again;) {
    Again = false;
    for (auto It = Pending.begin(); It != Pending.end();) {
      Instruction *I = *It;
      if (!all_of(I->users(), [I](User *U) { return U == I; })) {
        ++It;
        continue;
      }
      I->dropAllReferences();
      I->eraseFromParent();
      It = Pending.erase(It);
      Again = true;
    }
  }
  return true;
}

} // namespace

PreservedAnalyses HipFunctionPointerASPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  bool Changed = fixGlobalInitializers(M);
  Changed |= fixIndirectCalls(M);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
