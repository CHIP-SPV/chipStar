//===- HipLowerSwitch.cpp -------------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lowers canonicalized switch instructions with input condition having a
// "non-standard" integer bitwidth (e.g. i4) to bitwidth supported by
// SPIRV-LLVM-Translator tool. Related issue:
//
// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/1685
//
// (c) 2023 CHIP-SPV Developers
//===----------------------------------------------------------------------===//

#include "HipLowerSwitch.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#define PASS_NAME "hip-lower-switch"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

static bool IsSupportedSwitchCondWidth(unsigned BitWidth) {
  return BitWidth >= 8 && BitWidth <= 64 && isPowerOf2_32(BitWidth);
}

static bool lowerSwitch(SwitchInst *SI,
                        SmallPtrSetImpl<Instruction *> &EraseSet) {
  // The implementation only covers the following pattern:
  //
  //    %cond = trunc iX %src to iY
  //    switch iY %cond, ...
  //
  // And transforms it to:
  //
  //  %newCond = and iX %src, <(1 << Y) - 1>
  //  switch iX %newCond, ...

  Value *OldCond = SI->getCondition();
  Type *OldType = OldCond->getType();
  auto OldWidth = cast<IntegerType>(OldType)->getBitWidth();

  if (IsSupportedSwitchCondWidth(OldWidth))
    return false;

  LLVM_DEBUG(dbgs() << "Needs lowering: " << *SI << "\n";
             dbgs() << "  Cond: " << *OldCond << "\n");

  if (!isa<TruncInst>(OldCond)) {
    LLVM_DEBUG(dbgs() << "  Bail out: Unmatched pattern (didn't see trunc).");
    return false;
  }

  auto *NewCond = cast<TruncInst>(OldCond)->getOperand(0);
  unsigned NewWidth = cast<IntegerType>(NewCond->getType())->getBitWidth();
  if (!IsSupportedSwitchCondWidth(NewWidth)) {
    LLVM_DEBUG(
        dbgs() << "  Bail out: trunc's source bitwidth is also unsupported.");
    return false;
  }

  auto *NewCondTy = IntegerType::get(OldCond->getContext(), NewWidth);

  // Mask the wider condition so trunc semantic is not lost.
  // TODO: Consult computeKnownBits() for optimizing the mask instruction away.
  auto *Mask =
      ConstantInt::get(NewCondTy, cast<IntegerType>(OldType)->getBitMask());
  NewCond = BinaryOperator::CreateAnd(NewCond, Mask, "switch.mask", SI);

  SI->setCondition(NewCond);
  for (auto Case : SI->cases()) {
    const APInt &CaseVal = Case.getCaseValue()->getValue();
    Case.setValue(ConstantInt::get(SI->getContext(), CaseVal.zext(NewWidth)));
  }

  EraseSet.insert(cast<Instruction>(OldCond));

  return true;
}

static bool lowerSwitches(Function &F) {
  bool Changed = false;

  SmallPtrSet<Instruction *, 4> EraseSet;
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *SI = dyn_cast<SwitchInst>(&I))
        Changed = lowerSwitch(SI, EraseSet);

  for (auto *ToErase : EraseSet)
    if (ToErase->hasNUses(0))
      ToErase->eraseFromParent();
    else
      LLVM_DEBUG(
          dbgs() << "Could not erase unsupported instruction (has uses)\n";
          dbgs() << "Instruction: " << *ToErase << "\n";);

  return Changed;
}

PreservedAnalyses HipLowerSwitchPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  return lowerSwitches(F) ? PreservedAnalyses::none()
                          : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-lower-switch", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_NAME) {
                    FPM.addPass(HipLowerSwitchPass());
                    return true;
                  }
                  return false;
                });
          }};
}
