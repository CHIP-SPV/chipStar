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
#include <string>

using namespace llvm;

class HipIRSpirvValidationPass : public PassInfoMixin<HipIRSpirvValidationPass> {
public:
  explicit HipIRSpirvValidationPass(const std::string& StageMsg, bool EnableSPIRVValidation = false) 
    : StageMsg(StageMsg), EnableSPIRVValidation(EnableSPIRVValidation) {}
  
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }

  /// Convert LLVM IR module to SPIR-V binary programmatically (public for testing)
  std::vector<uint32_t> convertIRToSPIRV(Module &M);

private:
  std::string StageMsg;
  bool EnableSPIRVValidation;
  
  // IR validation methods
  bool runOptVerify(Module &M);
  
  // SPIR-V validation methods
  bool verifySPIRVBinary(const std::vector<uint32_t> &spirvBinary);
  bool isCompileTimeVerificationEnabled();
};

#endif // LLVM_PASSES_HIP_IR_SPIRV_VALIDATION_H 