//===- HipVerify.h -------------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Unified verification pass that runs IR verification, SPIR-V conversion,
// and SPIR-V validation with table summary output.
//
// (c) 2024 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_VERIFY_H
#define LLVM_PASSES_HIP_VERIFY_H

#include "llvm/IR/PassManager.h"
#include <vector>
#include <string>

namespace llvm {

class HipVerifyPass : public PassInfoMixin<HipVerifyPass> {
public:
  struct VerificationResult {
    std::string ModuleName;
    std::string PassName;
    bool IRValidatePass;
    std::string IRValidateError;
    bool SPIRVCompilePass;
    std::string SPIRVCompileError;
    bool SPIRVValidatePass;
    std::string SPIRVValidateError;
    // Tool paths used during verification
    std::string OptPath;
    std::string LLVMAsPath;
    std::string LLVMSpirvPath;
    std::string SpirvValPath;
  };

  explicit HipVerifyPass(const std::string& PassName = "HipVerify", 
                        bool PrintSummary = true) 
    : PassName(PassName), PrintSummary(PrintSummary) {}
  
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
  
  // Clear all collected results (call at start of pipeline)
  static void clearResults() { AllResults.clear(); }

private:
  std::string PassName;
  bool PrintSummary;
  
  // Verification methods
  static std::string getVerificationMode();
  bool isVerificationEnabled();
  bool runIRVerification(Module &M, VerificationResult &result);
  bool runSPIRVConversion(Module &M, std::vector<uint32_t> &spirvBinary, VerificationResult &result);
  bool runSPIRVValidation(const std::vector<uint32_t> &spirvBinary, VerificationResult &result);
  
  // SPIR-V conversion helpers
  std::vector<uint32_t> convertIRToSPIRV(Module &M, std::string &errorMsg, VerificationResult &result);
  
  // Function reordering for SPIR-V compliance
  void reorderFunctionsForSPIRV(Module &M);
  bool needsReordering(Module &M);
  bool isSPIRVTarget(Module &M);
  
  // Summary output
  void printSummaryTable(const VerificationResult &result);
  
  // Text wrapping helper
  static std::vector<std::string> wrapText(const std::string &text, size_t width);
  
  // Static table for collecting results across multiple modules
  static std::vector<VerificationResult> AllResults;
  static void printFinalSummary();
};

} // namespace llvm

#endif // LLVM_PASSES_HIP_VERIFY_H 