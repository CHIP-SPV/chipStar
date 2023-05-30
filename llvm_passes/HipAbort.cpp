//===- HipAbort.cpp -.-----------------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR pass to handle kernels with abort() calls.
//
// (c) 2023 CHIP-SPV developers
//     2022 Pekka Jääskeläinen / Parmance for Argonne National Laboratory
//===----------------------------------------------------------------------===//
//
// HIP supports an abort() that is specified as
// "...will terminate the process execution from inside the kernel."
// https://rocmdocs.amd.com/en/latest/Programming_Guides/Kernel_language.html
//
// We approximate this behavior with a global host-readable variable that is set
// when abort() is called. The runtime, after detecting the variable is set
// calls the process abort() immediately. This LLVM pass treats abort() calls
// so we do not execute any kernel code after calling them.
//
// This is handled by converting all function calls (that potentially call abort
// or _are_ abort, perhaps recursively) as following:
//
// func_call();
//
// ..is converted to...
//
// func_call(); if (__chipspv_abort_called) return; [poison?]
//
// Thus it forces the call tree be unwound similar to an exception in C++.
//
// Another option would have been to just busy loop in abort() after setting the
// flag and poll at host side for the abort flag. However, it then makes the
// device busy and reading the global variable might not work (since it
// currently requires kernel execution).
//
// Note: This doesn't handle all cases. Especially kernels with barrier calls
// likely won't work if there are diverging abort calls as some of the threads
// return early and might never reach the barrier due to a race condition.
// A proper all-cases covering implementation would be best done with
// SPIR-V/OpenCL support, which affects how barriers are treated.
//===----------------------------------------------------------------------===//

#include "HipAbort.h"

#include "../src/common.hh"

#include <set>
#include <iostream>

#define DEBUG_TYPE "hip-abort"

using InverseCallGraphNode = HipAbortPass::InverseCallGraphNode;
using AbortAttribute = HipAbortPass::AbortAttribute;
using CallRecord = CallGraphNode::CallRecord;

static StringRef getFnName(const Function *F) {
  if (!F)
    return "<error: nullptr>";
  if (!F->hasName())
    return "<error: no-name>";
  return F->getName();
}

/// Get corresponding inverted call graph node for the function 'F'.
InverseCallGraphNode *HipAbortPass::getInvertedCGNode(Function *F) {
  assert(F && "Key is nullptr!");
  auto *Node = InverseCallGraph[F];
  assert(Node && "Node was not found for a function.");
  return Node;
}

/// Get corresponding inverted call graph node for 'CGN'.
InverseCallGraphNode *
HipAbortPass::getInvertedCGNode(const CallGraphNode *CGN) {
  return getInvertedCGNode(CGN->getFunction());
}

/// Create or get already created node for inverted call graph for the regular
/// call graph node.
InverseCallGraphNode *
HipAbortPass::getOrCreateInvertedCGNode(const CallGraphNode *CGN) {
  auto *F = CGN->getFunction();
  if (!InverseCallGraph.count(F))
    InverseCallGraph[F] = new InverseCallGraphNode(CGN);
  return getInvertedCGNode(F);
}

/// Build an inverted call graph where the call edges are reversed. The produced
/// graph only includes defined functions.
void HipAbortPass::buildInvertedCallGraph(const CallGraph &CG) {
  for (auto &FnNodePair : CG) {
    auto *CGCaller = FnNodePair.second.get();
    if (CGCaller == CG.getExternalCallingNode() ||
        CGCaller == CG.getCallsExternalNode())
      continue;
    assert(CGCaller);

    auto *ICGCaller = getOrCreateInvertedCGNode(CGCaller);

    if (!CGCaller->size()) {
      // The function does not call anything, thus, won't abort either.
      ICGCaller->AbortAttr = AbortAttribute::WontAbort;
      continue;
    }

    for (auto &CGRecord : *CGCaller) {
      auto *CGCalleeNode = CGRecord.second;
      if (CGCalleeNode == CG.getCallsExternalNode())
        continue;
      auto *ICGCallee = getOrCreateInvertedCGNode(CGCalleeNode);
      ICGCallee->Callers.insert(ICGCaller);
    }
  }
}

/// Returns true if the 'CI' is a *direct* call to abort declaration.
static bool callsAbort(const CallInst *CI) {
  return CI->getCalledFunction() &&
         CI->getCalledFunction()->getName() == "__chipspv_abort";
}

/// Get CallInst recorded in 'CR' if it has one. Otherwise, return nullptr.
static CallInst *getCallInst(const CallRecord &CR) {
  // Get call value, peel off std::optional.
  if (!CR.first)
    return nullptr;
  // Peel off WeakTrackingVH. value should The be alive as module is not
  // modfied at this point.
  assert(CR.first->pointsToAliveValue());
  Value *V = static_cast<Value *>(*CR.first);
  // Must be a call instruction. Otherwise, something is very off if there are
  // invoke or callbr instructions in the device code.
  return cast<CallInst>(V);
}

static InverseCallGraphNode *popAny(std::set<InverseCallGraphNode *> &Set) {
  assert(Set.size() && "Can't extract an element from empty container!");
  auto EltIt = Set.begin();
  Set.erase(EltIt);
  return *EltIt;
}

void HipAbortPass::analyze(const CallGraph &CG) {
  std::set<InverseCallGraphNode *> WorkList;
  SmallVector<InverseCallGraphNode *> KernelNodes;

  // Find functions directly calling __chipspv_abort() function.
  for (auto &FnNodePair : CG) {
    auto *CGNode = FnNodePair.second.get();
    if (CGNode == CG.getExternalCallingNode() ||
        CGNode == CG.getCallsExternalNode())
      continue; // Only interested in functions with definition.
    auto *F = CGNode->getFunction();
    if (F->isDeclaration())
      continue;

    if (F->getCallingConv() == CallingConv::SPIR_KERNEL)
      KernelNodes.push_back(getInvertedCGNode(F));

    for (auto &CGRecord : *CGNode) {
      auto *CI = getCallInst(CGRecord);
      assert(CI);
      bool IndirectCall = !CI->getCalledFunction();
      LLVM_DEBUG({
        if (IndirectCall) {
          // Current analysis does not extend beyond indirect calls (which are
          // fortunately rare). Since they may potentially call the abort(),
          // mark the call's origin function with 'MayAbort'.
          dbgs() << "    Warning: an indirect call considered as may-abort\n"
                 << "Call origin: " << getFnName(CI->getParent()->getParent())
                 << "\n";
        }
      });

      if (callsAbort(CI) || IndirectCall) {
        WorkList.insert(getInvertedCGNode(F));
        break;
      }
    }
  }

  // Propagate may-abort attribute to callers.
  while (WorkList.size()) {
    InverseCallGraphNode *N = popAny(WorkList);
    assert(N->AbortAttr != AbortAttribute::MayAbort && "Infinite loop!");
    N->AbortAttr = AbortAttribute::MayAbort;
    for (auto *Caller : N->Callers)
      if (Caller->AbortAttr != AbortAttribute::MayAbort)
        WorkList.insert(Caller);
  }

  LLVM_DEBUG({
    dbgs() << "Kernels potentially calling abort():\n";
    for (auto *KN : KernelNodes)
      if (KN->mayCallAbort())
        dbgs() << "- " << getFnName(KN->getFunction()) << "\n";
  });

  for (auto *KN : KernelNodes)
    if (KN->mayCallAbort()) {
      AnyKernelMayCallAbort = true;
      break;
    }
}

/// True if the callee may abort. This method can be only called after
/// analyze().
bool HipAbortPass::mayCallAbort(const CallInst *CI) const {
  if (callsAbort(CI))
    return true;

  if (!CI->getCalledFunction())
    return true; // Current analysis does not extend beyond indirect calls.

  auto *Callee = CI->getCalledFunction();
  if (InverseCallGraph.count(Callee))
    return false; // Reaching this could also mean incomplete analysis.

  return InverseCallGraph.at(Callee)->mayCallAbort();
}

void HipAbortPass::processFunctions(Module &M) {
  if (!AnyKernelMayCallAbort)
    return;

  std::set<CallInst *> CallsToHandle;
  std::set<CallInst *> AbortCallsToHandle;
  for (auto &FnNodePair : InverseCallGraph) {
    auto *ICGNode = FnNodePair.second;
    if (!ICGNode->mayCallAbort())
      continue;

    auto *CGNode = ICGNode->OrigNode;
    for (auto &CGRecord : *CGNode) {
      auto *CI = getCallInst(CGRecord);
      assert(CI);
      if (callsAbort(CI)) {
        CallsToHandle.insert(CI);
        AbortCallsToHandle.insert(CI);
      } else if (mayCallAbort(CI))
        CallsToHandle.insert(CI);
    }
  }

  // Analysis concluded a kernel may call abort() - there should be something to
  // be processed.
  assert(CallsToHandle.size() || AbortCallsToHandle.size());

  auto *AbortFlag = M.getGlobalVariable(ChipDeviceAbortFlagName);
  auto &Ctx = M.getContext();
  auto *Int32Ty = IntegerType::get(Ctx, 32);
  auto *Int32OneConst = ConstantInt::get(Int32Ty, 1);

  // Use a single abort exit BB per function.
  std::map<Function *, BasicBlock *> AbortReturnBlocks;

  for (auto Call : CallsToHandle) {
    Instruction *InstrAfterCall = Call->getNextNonDebugInstruction();
    Function *Func = Call->getParent()->getParent();
    llvm::BasicBlock *OrigBB = InstrAfterCall->getParent();
    llvm::BasicBlock *FallThrough = OrigBB->splitBasicBlock(InstrAfterCall);
    llvm::BasicBlock *ReturnBlock = AbortReturnBlocks[Func];
    if (ReturnBlock == nullptr) {
      ReturnBlock = AbortReturnBlocks[Func] =
          BasicBlock::Create(Ctx, "abort_return_bb", Func);

      Type *RetTy = Func->getFunctionType()->getReturnType();
      ReturnInst::Create(Ctx,
                         RetTy->isVoidTy() ? nullptr : UndefValue::get(RetTy),
                         ReturnBlock);
    }

    InstrAfterCall = Call->getNextNonDebugInstruction();
    IRBuilder<> B(InstrAfterCall);
    LoadInst *FlagLoad =
        B.CreateLoad(AbortFlag->getValueType(), AbortFlag, true, "abort_flag");
    Value *FlagCmp = B.CreateICmpNE(
        FlagLoad, ConstantInt::get(AbortFlag->getValueType(), 0));

    // Replace the fallthrough terminator BR in the old BB.
    B.CreateCondBr(FlagCmp, ReturnBlock, FallThrough);

    OrigBB->getTerminator()->eraseFromParent();
  }

  for (auto Call : AbortCallsToHandle) {
    // Convert the calls to the abort() decl to stores to the abort flag.
    IRBuilder<> B(Call);
    B.CreateStore(Int32OneConst, AbortFlag, true);
    Call->eraseFromParent();
  }
}

/// Erase a variable used for signaling abort event. Return true if it was found
/// and removed.
static bool eraseAbortFlag(Module &M) {
  // Mark modules that do not call abort by just the global flag variable.
  // Ugly, but should allow avoiding the kernel call to check the global
  // variable.
  GlobalVariable *AbortFlag = M.getGlobalVariable(ChipDeviceAbortFlagName);
  if (AbortFlag != nullptr) {
    AbortFlag->replaceAllUsesWith(Constant::getNullValue(AbortFlag->getType()));
    AbortFlag->eraseFromParent();
    return true;
  }
  return false;
}

PreservedAnalyses HipAbortPass::run(Module &Mod, ModuleAnalysisManager &AM) {
  reset();

  // This is always present due to its being declared as externally visible.  We
  // are dealing with fully linked device code so nothing should link to the
  // symbol. If we can remove it, we remove __chipspv_abort() call too which
  // gives us a change to exit the pass early.
  auto *AssertFailF = Mod.getFunction("__assert_fail");
  if (AssertFailF && AssertFailF->hasNUses(0))
    AssertFailF->eraseFromParent();

  // The abort calls are made to undefined abort decl thus should not get
  // inlined.
  auto *AbortF = Mod.getFunction("__chipspv_abort");
  if (!AbortF) {
    return eraseAbortFlag(Mod) ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
  }
  if (AbortF->hasNUses(0)) {
    AbortF->eraseFromParent();
    eraseAbortFlag(Mod);
    return PreservedAnalyses::none();
  }

  assert(AbortF->isDeclaration() && "Expected to be a declaration!");

  CallGraph CG(Mod);
  buildInvertedCallGraph(CG);
  analyze(CG);
  if (AnyKernelMayCallAbort)
    processFunctions(Mod);
  else
    eraseAbortFlag(Mod);

  // __chipspv_abort is not needed anymore. Eliminate it so SPIR-V linker does
  // not get upset about undefined symbol.
  if (AbortF->hasNUses(0))
    AbortF->eraseFromParent();
  else {
    // Simpler to just add an empy definition to it and let optimizer handle it.
    auto *BB = BasicBlock::Create(Mod.getContext(), "entry", AbortF);
    ReturnInst::Create(Mod.getContext(), BB);
    assert(!AbortF->isDeclaration());
  }

  return PreservedAnalyses::none();
}
