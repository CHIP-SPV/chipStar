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
// Clang emits C++ vtables for SPIR-V targets with every slot in the global
// address space, so each virtual function slot is a constant expression
//
//   addrspacecast (ptr @_ZN7DerivedD2Ev to ptr addrspace(1))
//
// SPIR-V only permits casts from a named storage class into Generic, never the
// other way around, so the module is rejected by the translator with
//
//   InvalidModule: Invalid SPIR-V module: Casts from private/local/global
//   address space are allowed only to generic
//
// This pass rewrites such vtables so their slots hold generic (address space
// 4) pointers instead, which is the only address space a function pointer may
// legally live in, and then repairs the loads and call sites that read them.
// With that done SPV_INTEL_function_pointers is sufficient to translate and
// run device side virtual dispatch. See CHIP-SPV/chipStar#1373.
//
// The pass is deliberately not keyed on the Itanium '_ZTV' vtable name
// prefix. The trigger is semantic: any global whose initializer casts a
// Function into a non-generic address space. That also covers hand written
// dispatch tables and any other construct with the same shape.
//
// Only the element type of the table changes. The vptr, that is the pointer
// *to* the vtable which is stored inside the object, is an ordinary data
// pointer and legally stays in the global address space.
//
// A vtable also holds non-function data such as the offset-to-top and the RTTI
// descriptor. Retyping the whole homogeneous array moves those into generic as
// well, which is a global -> generic cast and therefore the legal direction.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipFunctionPointerAS.h"

#include "LLVMSPIRV.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Debug.h"

#include "PassPluginCompat.h"

#define DEBUG_TYPE "hip-function-pointer-as"

using namespace llvm;

namespace {

constexpr unsigned GenericAS = SPIRV_GENERIC_AS;

/// True for 'addrspacecast (ptr @SomeFunction to ptr addrspace(N))' with N not
/// being the generic address space. This is the exact construct the SPIR-V
/// translator rejects.
static bool isFunctionCastToNonGeneric(const Constant *C) {
  const auto *CE = dyn_cast<ConstantExpr>(C);
  if (!CE || CE->getOpcode() != Instruction::AddrSpaceCast)
    return false;
  if (!isa<Function>(CE->getOperand(0)))
    return false;
  return CE->getType()->getPointerAddressSpace() != GenericAS;
}

/// True if \p C is, or transitively contains, an illegal function pointer cast.
static bool containsFunctionCastToNonGeneric(const Constant *C,
                                             SmallPtrSetImpl<const Constant *> &Seen) {
  if (!C || !Seen.insert(C).second)
    return false;
  if (isFunctionCastToNonGeneric(C))
    return true;
  // Do not descend into other globals; they are visited on their own.
  if (isa<GlobalValue>(C))
    return false;
  for (const Use &U : C->operands())
    if (const auto *Op = dyn_cast<Constant>(&U))
      if (containsFunctionCastToNonGeneric(Op, Seen))
        return true;
  return false;
}

/// Rewrite \p C, which must be of some pointer type, so that its type becomes
/// 'ptr addrspace(4)'.
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

  // Rebuild an existing addrspacecast from its original operand rather than
  // stacking a second cast on top of an already illegal one.
  if (auto *CE = dyn_cast<ConstantExpr>(C))
    if (CE->getOpcode() == Instruction::AddrSpaceCast)
      return ConstantExpr::getAddrSpaceCast(CE->getOperand(0), NewTy);

  // Plain data operands, for example the RTTI descriptor slot, live in the
  // global address space. Global -> generic is the legal cast direction.
  return ConstantExpr::getAddrSpaceCast(C, NewTy);
}

/// Recursively rebuild \p C so that every function pointer it holds is in the
/// generic address space. Returns the rewritten constant, which may have a
/// different type than \p C, or \p C itself when nothing had to change.
static Constant *rewriteInitializer(Constant *C) {
  if (!C)
    return C;

  if (isFunctionCastToNonGeneric(C))
    return coercePtrToGeneric(C);

  // Nested aggregates: rewrite element wise.
  auto *CA = dyn_cast<ConstantAggregate>(C);
  if (!CA)
    return C;

  SmallVector<Constant *, 8> Elts;
  bool Changed = false;
  for (unsigned I = 0, E = CA->getNumOperands(); I < E; ++I) {
    Constant *Old = CA->getOperand(I);
    Constant *New = rewriteInitializer(Old);
    Changed |= New != Old;
    Elts.push_back(New);
  }
  if (!Changed)
    return C;

  if (isa<ConstantArray>(CA)) {
    // Arrays are homogeneous: once one slot moved to generic, every slot has
    // to follow, including the null and RTTI slots of a vtable.
    for (Constant *&E : Elts)
      if (E->getType()->isPointerTy())
        E = coercePtrToGeneric(E);
    assert(!Elts.empty());
    Type *EltTy = Elts.front()->getType();
    for (Constant *E : Elts)
      if (E->getType() != EltTy)
        return C; // Heterogeneous after rewriting; leave it alone.
    return ConstantArray::get(ArrayType::get(EltTy, Elts.size()), Elts);
  }

  if (auto *CS = dyn_cast<ConstantStruct>(CA)) {
    // Struct members keep their individual types, so use a literal struct with
    // the rewritten member types. Vtables are literal structs in practice.
    return ConstantStruct::getAnon(CS->getContext(), Elts,
                                   CS->getType()->isPacked());
  }

  return C;
}

/// Replace \p Old with a new global holding \p NewInit. With opaque pointers
/// the address of the global has the same LLVM type regardless of the value
/// type, so all users can simply be redirected.
static GlobalVariable *replaceGlobalInitializer(Module &M, GlobalVariable *Old,
                                                Constant *NewInit) {
  auto *New = new GlobalVariable(
      M, NewInit->getType(), Old->isConstant(), Old->getLinkage(), NewInit, "",
      nullptr, Old->getThreadLocalMode(), Old->getAddressSpace(),
      Old->isExternallyInitialized());

  New->copyAttributesFrom(Old);
  New->setAlignment(Old->getAlign());
  New->setComdat(Old->getComdat());
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  Old->getAllMetadata(MDs);
  for (const auto &MD : MDs)
    New->addMetadata(MD.first, *MD.second);

  // A constant getelementptr stores its source element type explicitly, so
  // replaceAllUsesWith would leave GEPs that still claim the *old* value type
  // while pointing at the retyped global. Classes with virtual bases hit this:
  // the VTT (_ZTT...) holds one 'getelementptr inrange(...) ({...}, ptr @_ZTC...)'
  // per construction vtable, and once a _ZTC table is retyped the SPIR-V writer
  // aborts in transConstantUse() on the disagreement. Rebuild those GEPs against
  // the new value type first, keeping indices, nowrap flags and inrange.
  Type *OldValTy = Old->getValueType();
  Type *NewValTy = New->getValueType();
  SmallVector<ConstantExpr *, 8> GEPUsers;
  for (User *U : Old->users())
    if (auto *CE = dyn_cast<ConstantExpr>(U))
      if (CE->getOpcode() == Instruction::GetElementPtr &&
          cast<GEPOperator>(CE)->getSourceElementType() == OldValTy)
        GEPUsers.push_back(CE);

  for (ConstantExpr *CE : GEPUsers) {
    auto *GEP = cast<GEPOperator>(CE);
    SmallVector<Value *, 4> Idxs(CE->op_begin() + 1, CE->op_end());
    Constant *NewCE = ConstantExpr::getGetElementPtr(
        NewValTy, New, Idxs, GEP->getNoWrapFlags(), GEP->getInRange());
    CE->replaceAllUsesWith(NewCE);
  }

  Old->replaceAllUsesWith(New);
  std::string Name = Old->getName().str();
  Old->eraseFromParent();
  New->setName(Name);
  return New;
}

/// Produce a value equivalent to \p V but typed 'ptr addrspace(4)'.
///
/// The interesting case by far is a load of a vtable slot. Its result type has
/// to change, because after the table itself has been retyped the memory now
/// holds a generic pointer and a load claiming to read a global pointer would
/// translate to a SPIR-V OpLoad whose result type disagrees with the pointee.
/// Inserting a cast after the load would not help for the same reason.
///
/// Anything not understood falls back to an inserted addrspacecast, which is
/// always valid when the source is a named storage class.
///
/// \p Visited breaks cycles, which phis of function pointers can form.
static Value *makeCalleeGeneric(Value *V, Instruction *InsertBefore,
                                SmallSetVector<Instruction *, 8> &Dead,
                                SmallPtrSetImpl<Value *> &Visited) {
  auto *PtrTy = dyn_cast<PointerType>(V->getType());
  if (!PtrTy || PtrTy->getAddressSpace() == GenericAS)
    return V;

  auto *GenericPtrTy = PointerType::get(V->getContext(), GenericAS);

  if (!Visited.insert(V).second) {
    // Already being rewritten further up the recursion; fall back to a cast.
    IRBuilder<> B(InsertBefore);
    return B.CreateAddrSpaceCast(V, GenericPtrTy);
  }

  // Look through a cast back into a named storage class: its source is already
  // what we want. Several calls sharing one slot load reach this, because the
  // first call rewrites the load and leaves the others reading through such a
  // compensating cast.
  if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V))
    if (ASC->getSrcAddressSpace() == GenericAS) {
      Dead.insert(ASC);
      return ASC->getPointerOperand();
    }

  if (auto *LI = dyn_cast<LoadInst>(V)) {
    IRBuilder<> B(LI);
    Value *Ptr = LI->getPointerOperand();

    // The address of the slot is normally computed by a getelementptr whose
    // source element type is the old slot type. Both are pointers of the same
    // size so the byte offset is unchanged, but keeping the types consistent
    // avoids handing the SPIR-V translator a load that disagrees with the
    // pointee type of its operand.
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr))
      if (auto *SrcPtrTy = dyn_cast<PointerType>(GEP->getSourceElementType()))
        if (SrcPtrTy->getAddressSpace() != GenericAS) {
          SmallVector<Value *, 4> Idx(GEP->idx_begin(), GEP->idx_end());
          IRBuilder<> GB(GEP);
          auto *NewGEP = cast<GetElementPtrInst>(GB.CreateGEP(
              GenericPtrTy, GEP->getPointerOperand(), Idx, GEP->getName()));
#if LLVM_VERSION_MAJOR >= 19
          NewGEP->setNoWrapFlags(GEP->getNoWrapFlags());
#else
          NewGEP->setIsInBounds(GEP->isInBounds());
#endif
          NewGEP->setDebugLoc(GEP->getDebugLoc());
          Ptr = NewGEP;
          GEP->replaceAllUsesWith(NewGEP);
          Dead.insert(GEP);
        }

    auto *NewLI = B.CreateAlignedLoad(GenericPtrTy, Ptr, LI->getAlign(),
                                      LI->isVolatile(), LI->getName());
    NewLI->setOrdering(LI->getOrdering());
    NewLI->setSyncScopeID(LI->getSyncScopeID());
    NewLI->copyMetadata(*LI);
    NewLI->setDebugLoc(LI->getDebugLoc());

    // Any remaining reader of the old load still wants the original address
    // space. Generic -> global is a legal narrowing cast in SPIR-V.
    if (!LI->use_empty()) {
      auto *Back = cast<Instruction>(B.CreateAddrSpaceCast(NewLI, PtrTy));
      LI->replaceAllUsesWith(Back);
      Dead.insert(Back);
    }
    Dead.insert(LI);
    return NewLI;
  }

  if (auto *Phi = dyn_cast<PHINode>(V)) {
    // Rebuild the phi in the generic address space so that a virtual call
    // reached from several predecessors still sees a single generic value.
    // Each incoming value is rewritten in turn, so that a slot load feeding
    // the phi is retyped rather than merely cast.
    IRBuilder<> B(Phi);
    auto *NewPhi = B.CreatePHI(GenericPtrTy, Phi->getNumIncomingValues(),
                               Phi->getName());
    NewPhi->setDebugLoc(Phi->getDebugLoc());
    for (unsigned I = 0, E = Phi->getNumIncomingValues(); I < E; ++I) {
      BasicBlock *BB = Phi->getIncomingBlock(I);
      Value *In = Phi->getIncomingValue(I);
      Value *NewIn;
      if (auto *CIn = dyn_cast<Constant>(In))
        NewIn = coercePtrToGeneric(CIn);
      else
        NewIn = makeCalleeGeneric(In, BB->getTerminator(), Dead, Visited);
      NewPhi->addIncoming(NewIn, BB);
    }
    Dead.insert(Phi);
    return NewPhi;
  }

  if (auto *C = dyn_cast<Constant>(V))
    return coercePtrToGeneric(C);

  IRBuilder<> B(InsertBefore);
  return B.CreateAddrSpaceCast(V, GenericPtrTy);
}

/// Rebuild \p CB so that its callee operand is \p NewCallee. LLVM encodes the
/// address space of an indirect callee in the call itself, so the instruction
/// has to be recreated rather than patched in place.
static void rewriteCallCallee(CallBase *CB, Value *NewCallee) {
  SmallVector<Value *, 8> Args(CB->args());
  SmallVector<OperandBundleDef, 2> Bundles;
  CB->getOperandBundlesAsDefs(Bundles);

  CallBase *NewCB;
  if (auto *CI = dyn_cast<CallInst>(CB)) {
    auto *NewCI =
        CallInst::Create(CB->getFunctionType(), NewCallee, Args, Bundles, "", CB);
    NewCI->setTailCallKind(CI->getTailCallKind());
    NewCB = NewCI;
  } else if (auto *II = dyn_cast<InvokeInst>(CB)) {
    NewCB = InvokeInst::Create(CB->getFunctionType(), NewCallee,
                               II->getNormalDest(), II->getUnwindDest(), Args,
                               Bundles, "", CB);
  } else {
    return;
  }

  NewCB->setCallingConv(CB->getCallingConv());
  NewCB->setAttributes(CB->getAttributes());
  NewCB->copyMetadata(*CB);
  NewCB->setDebugLoc(CB->getDebugLoc());
  // Fast-math flags, 'contract' in particular, are not carried by metadata.
  if (isa<FPMathOperator>(CB) && isa<FPMathOperator>(NewCB))
    NewCB->setFastMathFlags(CB->getFastMathFlags());

  NewCB->takeName(CB);
  CB->replaceAllUsesWith(NewCB);
  CB->eraseFromParent();
}

/// Retype vtable-like globals. Returns true if the module changed.
static bool fixGlobalInitializers(Module &M) {
  SmallVector<GlobalVariable *, 8> Candidates;
  for (GlobalVariable &GV : M.globals()) {
    if (!GV.hasInitializer())
      continue;
    SmallPtrSet<const Constant *, 16> Seen;
    if (containsFunctionCastToNonGeneric(GV.getInitializer(), Seen))
      Candidates.push_back(&GV);
  }

  bool Changed = false;
  for (GlobalVariable *GV : Candidates) {
    Constant *NewInit = rewriteInitializer(GV->getInitializer());
    if (NewInit == GV->getInitializer())
      continue;
    LLVM_DEBUG(dbgs() << "hip-function-pointer-as: retyping " << GV->getName()
                      << "\n");
    replaceGlobalInitializer(M, GV, NewInit);
    Changed = true;
  }
  return Changed;
}

/// Repair indirect calls whose callee is not in the generic address space.
static bool fixIndirectCalls(Module &M) {
  SmallVector<CallBase *, 8> Calls;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (!CB || !CB->isIndirectCall())
          continue;
        Value *Callee = CB->getCalledOperand();
        unsigned AS = Callee->getType()->getPointerAddressSpace();
        // The program address space is what a plain 'call' uses and is always
        // fine; generic is what we are aiming for.
        if (AS == GenericAS || AS == M.getDataLayout().getProgramAddressSpace())
          continue;
        Calls.push_back(CB);
      }

  if (Calls.empty())
    return false;

  // A set so that an instruction reached twice, which happens when several
  // calls share one slot load, is not erased twice.
  SmallSetVector<Instruction *, 8> Dead;
  for (CallBase *CB : Calls) {
    SmallPtrSet<Value *, 8> Visited;
    Value *NewCallee =
        makeCalleeGeneric(CB->getCalledOperand(), CB, Dead, Visited);
    if (NewCallee != CB->getCalledOperand())
      rewriteCallCallee(CB, NewCallee);
  }

  // Drop the originals now that nothing refers to them any more.
  for (Instruction *I : reverse(Dead))
    if (I->use_empty())
      I->eraseFromParent();

  return true;
}

} // namespace

PreservedAnalyses HipFunctionPointerASPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  bool Changed = fixGlobalInitializers(M);
  Changed |= fixIndirectCalls(M);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
