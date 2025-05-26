//===- HipFinalIRVerification.h ------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Combined IR and SPIR-V validation pass that runs before and after all HIP passes.
// Validates IR using opt -verify and converts/validates SPIR-V at the final stage.
//
// (c) 2024 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_IR_SPIRV_VALIDATION_H
#define LLVM_PASSES_HIP_IR_SPIRV_VALIDATION_H

#include "llvm/IR/PassManager.h"
#include <vector>

using namespace llvm;

enum class ValidationStage {
  Initial,  // Before HIP passes - IR validation only
  Final     // After HIP passes - IR validation + SPIR-V validation
};

class HipIRSpirvValidationPass : public PassInfoMixin<HipIRSpirvValidationPass> {
public:
  explicit HipIRSpirvValidationPass(ValidationStage Stage) : Stage(Stage) {}
  
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }

  /// Convert LLVM IR module to SPIR-V binary programmatically (public for testing)
  std::vector<uint32_t> convertIRToSPIRV(Module &M);

private:
  ValidationStage Stage;
  std::string StageMsg;
  
  // IR validation methods
  bool runOptVerify(Module &M);
  
  // SPIR-V validation methods
  bool verifySPIRVBinary(const std::vector<uint32_t> &spirvBinary);
  bool isCompileTimeVerificationEnabled();
  
  const char* getStageString() const {
    return Stage == ValidationStage::Initial ? "before" : "after";
  }
};

// Convenience aliases for the two validation stages
using HipInitialIRSpirvValidationPass = HipIRSpirvValidationPass;
using HipFinalIRSpirvValidationPass = HipIRSpirvValidationPass;

// Factory functions for creating the passes
inline HipInitialIRSpirvValidationPass createHipInitialIRSpirvValidationPass() {
  return HipInitialIRSpirvValidationPass(ValidationStage::Initial);
}

inline HipFinalIRSpirvValidationPass createHipFinalIRSpirvValidationPass() {
  return HipFinalIRSpirvValidationPass(ValidationStage::Final);
}

#endif // LLVM_PASSES_HIP_IR_SPIRV_VALIDATION_H 