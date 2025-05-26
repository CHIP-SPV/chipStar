//===- HipFinalIRVerification.h ------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Simple IR verification pass that runs before and after all HIP passes
// by invoking opt -verify to ensure the IR is valid.
//
// (c) 2024 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_FINAL_IR_VERIFICATION_H
#define LLVM_PASSES_HIP_FINAL_IR_VERIFICATION_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

enum class VerificationStage {
  Initial,  // Before HIP passes
  Final     // After HIP passes
};

class HipIRVerificationPass : public PassInfoMixin<HipIRVerificationPass> {
public:
  explicit HipIRVerificationPass(VerificationStage Stage) : Stage(Stage) {}
  
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  VerificationStage Stage;
  
  bool runOptVerify(Module &M);
  
  const char* getStageString() const {
    return Stage == VerificationStage::Initial ? "before" : "after";
  }
};

// Convenience aliases for the two verification stages
using HipInitialIRVerificationPass = HipIRVerificationPass;
using HipFinalIRVerificationPass = HipIRVerificationPass;

// Factory functions for creating the passes
inline HipInitialIRVerificationPass createHipInitialIRVerificationPass() {
  return HipInitialIRVerificationPass(VerificationStage::Initial);
}

inline HipFinalIRVerificationPass createHipFinalIRVerificationPass() {
  return HipFinalIRVerificationPass(VerificationStage::Final);
}

#endif // LLVM_PASSES_HIP_FINAL_IR_VERIFICATION_H 