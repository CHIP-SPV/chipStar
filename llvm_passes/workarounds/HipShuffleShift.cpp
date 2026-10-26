//===- HipShuffleShift.cpp ------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Rewrites constant-offset sub-group shuffles as shifts.
//
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipShuffleShift.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <cstdlib>
#include <optional>
#include <string>
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {
// An OpenCL work-item query, e.g. _Z12get_local_idj.
static bool isWorkItemQuery(const Value *V, bool SizeOnly = false) {
  auto *CI = dyn_cast<CallInst>(V);
  Function *F = CI ? CI->getCalledFunction() : nullptr;
  if (!F || !F->isDeclaration() || CI->hasOperandBundles())
    return false;
  StringRef N = F->getName();
  if (!N.consume_front("_Z"))
    return false;
  N = N.drop_while([](char C) { return isDigit(C); });
  if (SizeOnly)
    return N.starts_with("get_") && N.contains("size");
  return N.starts_with("get_global_") || N.starts_with("get_local_") ||
         N.starts_with("get_group_id") || N.starts_with("get_num_groups") ||
         N.starts_with("get_enqueued_local_size") ||
         N.starts_with("get_sub_group_") ||
         N.starts_with("get_max_sub_group_size") ||
         N.starts_with("get_num_sub_groups");
}

// True if V can be recomputed at the function entry: it reads only
// work-item queries, constants and arguments, and cannot divide by zero.
// Sets D to the first constant added to or subtracted from a value.
static bool isHoistable(Value *V, int64_t &D, SmallPtrSet<Value *, 16> &Seen,
                        int Depth = 0) {
  if (isa<Constant>(V) || isa<Argument>(V) || Seen.contains(V))
    return true;
  auto *I = dyn_cast<Instruction>(V);
  if (!I || Depth > 16)
    return false;
  Seen.insert(V);
  if (isa<CallInst>(I))
    return isWorkItemQuery(I) &&
           all_of(cast<CallInst>(I)->args(), [](Value *A) {
             return isa<ConstantInt>(A);
           });
  if (I->isIntDivRem() || I->isShift()) {
    // No division by zero, no shift past the width.
    Value *Op = I->getOperand(1);
    auto *C = dyn_cast<ConstantInt>(Op);
    if (isa<ZExtInst, SExtInst>(Op))
      Op = cast<CastInst>(Op)->getOperand(0);
    bool Signed = I->getOpcode() == Instruction::SDiv ||
                  I->getOpcode() == Instruction::SRem;
    if (I->isShift()
            ? !C || C->getValue().uge(I->getType()->getIntegerBitWidth())
            : !(C && (Signed ? C->getValue().isStrictlyPositive()
                             : !C->isZero())) &&
                  !isWorkItemQuery(Op, true))
      return false;
  } else if (!isa<BinaryOperator, ZExtInst, SExtInst, TruncInst, ICmpInst,
                 SelectInst>(I)) {
    return false;
  }
  if (auto *C = dyn_cast<ConstantInt>(I->getOperand(I->getNumOperands() - 1));
      C && !D && !C->isZero() && C->getValue().getSignificantBits() <= 32 &&
      (I->getOpcode() == Instruction::Add ||
       I->getOpcode() == Instruction::Sub))
    D = I->getOpcode() == Instruction::Add ? C->getSExtValue()
                                           : -C->getSExtValue();
  return all_of(I->operands(), [&](Value *Op) {
    return isHoistable(Op, D, Seen, Depth + 1);
  });
}

// Copies a hoistable V to before At, each instruction once.
static Value *hoist(Value *V, Instruction *At,
                    DenseMap<Value *, Value *> &Copies) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return V;
  if (Value *Copy = Copies.lookup(I))
    return Copy;
  Instruction *New = I->clone();
  for (Use &Op : New->operands())
    if (!isa<Function>(Op))
      Op = hoist(Op, At, Copies);
  New->insertBefore(At->getIterator());
  return Copies[I] = New;
}

// V for sub-group local id L and max sub-group size M, if V depends on
// nothing else.
using EvalCache = DenseMap<Value *, std::optional<APInt>>;
static std::optional<APInt> evalUncached(Value *V, unsigned L, unsigned M,
                                         EvalCache &Cache, int Depth);
static std::optional<APInt> eval(Value *V, unsigned L, unsigned M,
                                 EvalCache &Cache, int Depth = 0) {
  if (auto *C = dyn_cast<ConstantInt>(V))
    return C->getValue();
  if (auto It = Cache.find(V); It != Cache.end())
    return It->second;
  auto R = evalUncached(V, L, M, Cache, Depth);
  Cache[V] = R;
  return R;
}
static std::optional<APInt> evalUncached(Value *V, unsigned L, unsigned M,
                                         EvalCache &Cache, int Depth) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I || Depth > 32 || !I->getType()->isIntegerTy())
    return std::nullopt;
  unsigned Bits = I->getType()->getIntegerBitWidth();
  if (auto *CI = dyn_cast<CallInst>(I)) {
    Function *F = CI->getCalledFunction();
    if (!F || CI->hasOperandBundles())
      return std::nullopt;
    if (F->getName() == "_Z22get_sub_group_local_idv")
      return APInt(Bits, L);
    if (F->getName() == "_Z22get_max_sub_group_sizev")
      return APInt(Bits, M);
    return std::nullopt;
  }
  SmallVector<APInt, 3> Ops;
  for (Value *Op : I->operands()) {
    auto A = eval(Op, L, M, Cache, Depth + 1);
    if (!A)
      return std::nullopt;
    Ops.push_back(*A);
  }
  if (isa<SelectInst>(I))
    return Ops[0].isOne() ? Ops[1] : Ops[2];
  if (auto *Cmp = dyn_cast<ICmpInst>(I))
    return APInt(1, ICmpInst::compare(Ops[0], Ops[1], Cmp->getPredicate()));
  switch (I->getOpcode()) {
  case Instruction::ZExt:
    return Ops[0].zext(Bits);
  case Instruction::SExt:
    return Ops[0].sext(Bits);
  case Instruction::Trunc:
    return Ops[0].trunc(Bits);
  case Instruction::Add:
    return Ops[0] + Ops[1];
  case Instruction::Sub:
    return Ops[0] - Ops[1];
  case Instruction::Mul:
    return Ops[0] * Ops[1];
  case Instruction::And:
    return Ops[0] & Ops[1];
  case Instruction::Or:
    return Ops[0] | Ops[1];
  case Instruction::Xor:
    return Ops[0] ^ Ops[1];
  case Instruction::Shl:
    return Ops[1].uge(Bits) ? std::nullopt : std::optional(Ops[0].shl(Ops[1]));
  case Instruction::LShr:
    return Ops[1].uge(Bits) ? std::nullopt : std::optional(Ops[0].lshr(Ops[1]));
  case Instruction::AShr:
    return Ops[1].uge(Bits) ? std::nullopt : std::optional(Ops[0].ashr(Ops[1]));
  case Instruction::UDiv:
    return Ops[1].isZero() ? std::nullopt : std::optional(Ops[0].udiv(Ops[1]));
  case Instruction::URem:
    return Ops[1].isZero() ? std::nullopt : std::optional(Ops[0].urem(Ops[1]));
  case Instruction::SDiv:
    return Ops[1].isZero() ? std::nullopt : std::optional(Ops[0].sdiv(Ops[1]));
  case Instruction::SRem:
    return Ops[1].isZero() ? std::nullopt : std::optional(Ops[0].srem(Ops[1]));
  default:
    return std::nullopt;
  }
}

// D if, for every max sub-group size the native module accepts, each lane L
// reads lane L, L + D, or L + D wrapped by the sub-group size.
static std::optional<int> provenShift(Value *Idx) {
  std::optional<int> D;
  for (unsigned M : {32u, 8u, 16u, 64u})
    for (unsigned L = 0; L < M; L++) {
      EvalCache Cache;
      auto I = eval(Idx, L, M, Cache);
      if (!I || I->getBitWidth() != 32 || I->uge(4 * M))
        return std::nullopt;
      int Diff = (int)I->getZExtValue() - (int)L;
      if (!D && Diff)
        D = std::abs(Diff - (int)M) < std::abs(Diff) ? Diff - (int)M : Diff;
      if (Diff && Diff != *D && Diff != *D - (int)M && Diff != *D + (int)M)
        return std::nullopt;
    }
  return D;
}

static CallInst *call(Module &M, StringRef Name, Type *Ret,
                      ArrayRef<Value *> Args, BasicBlock::iterator At) {
  SmallVector<Type *, 3> Tys;
  for (Value *A : Args)
    Tys.push_back(A->getType());
  FunctionCallee F =
      M.getOrInsertFunction(Name, FunctionType::get(Ret, Tys, false));
  cast<Function>(F.getCallee())->setCallingConv(CallingConv::SPIR_FUNC);
  cast<Function>(F.getCallee())->addFnAttr(Attribute::Convergent);
  auto *CI = CallInst::Create(F, Args, "", At);
  CI->setCallingConv(CallingConv::SPIR_FUNC);
  return CI;
}
} // namespace

PreservedAnalyses HipShuffleShiftPass::run(Function &F,
                                           FunctionAnalysisManager &AM) {
  SmallVector<CallInst *, 8> Sites;
  for (Instruction &I : instructions(F)) {
    // The calls this adds would mix uncontrolled with controlled convergence.
    if (auto *CB = dyn_cast<CallBase>(&I);
        CB && CB->getOperandBundle(LLVMContext::OB_convergencectrl))
      return PreservedAnalyses::all();
    if (auto *CI = dyn_cast<CallInst>(&I); CI && CI->getCalledFunction() &&
                                           !CI->hasOperandBundles() &&
                                           !CI->isMustTailCall())
      if (StringRef N = CI->getCalledFunction()->getName();
          N.size() == 23 && N.starts_with("_Z17sub_group_shuffle") &&
          StringRef("ijlmf").contains(N[21]))
        Sites.push_back(CI);
  }

  Module &M = *F.getParent();
  Type *I32 = Type::getInt32Ty(M.getContext());
  Instruction *Entry = &*F.getEntryBlock().getFirstInsertionPt();
  bool Changed = false;
  for (CallInst *CI : Sites) {
    Type *Ty = CI->getType();
    std::string Name = "__chip_sg_shift_";
    Name += CI->getCalledFunction()->getName()[21];
    // The index is a shift by construction: no test needed.
    if (auto PD = provenShift(CI->getArgOperand(1))) {
      CallInst *Shift =
          call(M, Name, Ty,
               {CI->getArgOperand(0), CI->getArgOperand(1),
                ConstantInt::get(I32, *PD, true)},
               CI->getIterator());
      CI->replaceAllUsesWith(Shift);
      CI->eraseFromParent();
      Changed = true;
      continue;
    }
    int64_t D = 0;
    SmallPtrSet<Value *, 16> Seen;
    if (!isHoistable(CI->getArgOperand(1), D, Seen) || !D || D != (int32_t)D)
      continue;
    DenseMap<Value *, Value *> Copies;
    // Frozen: lanes that never reach the shuffle may compute poison here.
    Value *Idx = new FreezeInst(hoist(CI->getArgOperand(1), Entry, Copies),
                                "", Entry->getIterator());
    Value *DV = ConstantInt::get(I32, D, true);
    auto *Ok = new ICmpInst(
        Entry->getIterator(), ICmpInst::ICMP_NE,
        call(M, "__chip_sg_shift_ok", I32, {Idx, DV}, Entry->getIterator()),
        ConstantInt::get(I32, 0));
    Instruction *Then, *Else;
    SplitBlockAndInsertIfThenElse(Ok, CI->getIterator(), &Then, &Else);
    BasicBlock *Tail = CI->getParent();
    CI->moveBefore(Else->getIterator());
    CallInst *Fast =
        call(M, Name, Ty, {CI->getArgOperand(0), CI->getArgOperand(1), DV},
             Then->getIterator());
    PHINode *Phi = PHINode::Create(Ty, 2, "", Tail->begin());
    CI->replaceAllUsesWith(Phi);
    Phi->addIncoming(Fast, Then->getParent());
    Phi->addIncoming(CI, Else->getParent());
    Changed = true;
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
