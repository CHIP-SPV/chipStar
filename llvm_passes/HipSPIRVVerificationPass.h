//===- HipSPIRVVerificationPass.h ----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM pass that converts IR to SPIR-V and verifies the result at compile time.
// This pass runs at the very end of the HIP pass pipeline to catch SPIR-V
// issues during compilation rather than at runtime.
//
// (c) 2024 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_SPIRV_VERIFICATION_PASS_H
#define LLVM_PASSES_HIP_SPIRV_VERIFICATION_PASS_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

class HipSPIRVVerificationPass : public PassInfoMixin<HipSPIRVVerificationPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }

  /// Convert LLVM IR module to SPIR-V binary programmatically (public for testing)
  std::vector<uint32_t> convertIRToSPIRV(Module &M);

private:
  /// Verify the SPIR-V binary using chipStar verification infrastructure
  bool verifySPIRVBinary(const std::vector<uint32_t> &spirvBinary);
  
  /// Check if compile-time SPIR-V verification is enabled
  bool isCompileTimeVerificationEnabled();
};

#endif // LLVM_PASSES_HIP_SPIRV_VERIFICATION_PASS_H 