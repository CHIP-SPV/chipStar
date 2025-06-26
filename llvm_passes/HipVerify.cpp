//===- HipVerify.cpp -----------------------------------------------------===//
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

#include "HipVerify.h"
#include "chipStarConfig.hh"

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Verifier.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Support/Format.h"

#include <cstdlib>
#include <string>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <optional>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "hip-verify"

// Static member initialization
std::vector<HipVerifyPass::VerificationResult> HipVerifyPass::AllResults;

PreservedAnalyses HipVerifyPass::run(Module &M, ModuleAnalysisManager &AM) {
  VerificationResult result;
  
  // Get module name
  result.ModuleName = M.getName().str();
  if (result.ModuleName.empty()) {
    result.ModuleName = M.getSourceFileName();
  }
  if (result.ModuleName.empty()) {
    result.ModuleName = "<unknown>";
  }
  
  result.PassName = PassName;
  
  // Check if verification is enabled
  std::string verifyMode = getVerificationMode();
  if (verifyMode == "off") {
    LLVM_DEBUG(dbgs() << "HipVerify: Verification disabled by environment variable\n");
    return PreservedAnalyses::all();
  }
  
  LLVM_DEBUG(dbgs() << "HipVerify: Starting verification for module " << result.ModuleName << "\n");
  
  // Step 1: IR Verification
  result.IRValidatePass = runIRVerification(M, result);
  
  // Step 2: SPIR-V Conversion (always attempt, even if IR validation failed)
  std::vector<uint32_t> spirvBinary;
  result.SPIRVCompilePass = runSPIRVConversion(M, spirvBinary, result);
  
  // Step 3: SPIR-V Validation (only if conversion succeeded)
  if (result.SPIRVCompilePass && !spirvBinary.empty()) {
    result.SPIRVValidatePass = runSPIRVValidation(spirvBinary, result);
  } else {
    result.SPIRVValidatePass = false;
    if (result.SPIRVCompileError.empty()) {
      result.SPIRVValidateError = "N/A";
    } else {
      result.SPIRVValidateError = "N/A";
    }
  }
  
  // Store result for final summary
  AllResults.push_back(result);
  
  // Print summary based on mode - this will be called after each pass
  // but the printFinalSummary function will handle the mode logic
  if (PrintSummary) {
    printFinalSummary();
  }
  
  return PreservedAnalyses::all();
}

std::string HipVerifyPass::getVerificationMode() {
  // Check environment variable
  const char* env = std::getenv("CHIP_VERIFY_MODE");
  if (!env) {
    // Default to "failures" if not specified
    return "failures";
  }
  
  std::string value(env);
  std::transform(value.begin(), value.end(), value.begin(), ::tolower);
  
  if (value == "off" || value == "0" || value == "false" || value == "no") {
    return "off";
  } else if (value == "all" || value == "always" || value == "1" || value == "true" || value == "yes") {
    return "all";
  } else if (value == "failures" || value == "fail" || value == "errors") {
    return "failures";
  } else {
    // Default to "failures" for unknown values
    return "failures";
  }
}

bool HipVerifyPass::isVerificationEnabled() {
  // Keep for backward compatibility, but now just checks if mode is not "off"
  return getVerificationMode() != "off";
}

bool HipVerifyPass::runIRVerification(Module &M, VerificationResult &result) {
  LLVM_DEBUG(dbgs() << "HipVerify: Running IR verification\n");
  
  std::string errorStr;
  raw_string_ostream errorStream(errorStr);
  
  bool passed = !llvm::verifyModule(M, &errorStream);
  errorStream.flush();
  
  result.IRValidatePass = passed;
  if (!passed) {
    result.IRValidateError = errorStr;
    // Remove newlines - line wrapping will handle display
    std::replace(result.IRValidateError.begin(), result.IRValidateError.end(), '\n', ' ');
  }
  
  LLVM_DEBUG(dbgs() << "HipVerify: IR verification " << (passed ? "PASSED" : "FAILED") << "\n");
  return passed;
}

bool HipVerifyPass::runSPIRVConversion(Module &M, std::vector<uint32_t> &spirvBinary, 
                                       VerificationResult &result) {
  LLVM_DEBUG(dbgs() << "HipVerify: Attempting SPIR-V conversion\n");
  
  std::string errorMsg;
  spirvBinary = convertIRToSPIRV(M, errorMsg);
  
  bool passed = !spirvBinary.empty();
  result.SPIRVCompilePass = passed;
  
  if (!passed) {
    result.SPIRVCompileError = errorMsg;
    // Remove newlines - line wrapping will handle display
    std::replace(result.SPIRVCompileError.begin(), result.SPIRVCompileError.end(), '\n', ' ');
  }
  
  LLVM_DEBUG(dbgs() << "HipVerify: SPIR-V conversion " << (passed ? "PASSED" : "FAILED") << "\n");
  return passed;
}

bool HipVerifyPass::runSPIRVValidation(const std::vector<uint32_t> &spirvBinary, 
                                       VerificationResult &result) {
  LLVM_DEBUG(dbgs() << "HipVerify: Running SPIR-V validation\n");
  
  // Basic SPIR-V validation
  if (spirvBinary.size() < 5) {
    result.SPIRVValidatePass = false;
    result.SPIRVValidateError = "Binary too small";
    return false;
  }
  
  // Check magic number
  const uint32_t SPIRVMagicNumber = 0x07230203;
  if (spirvBinary[0] != SPIRVMagicNumber) {
    result.SPIRVValidatePass = false;
    result.SPIRVValidateError = "Invalid magic number";
    return false;
  }
  
  // Write SPIR-V to temporary file for spirv-val
  SmallString<256> SPVTempFile;
  std::error_code EC;
  EC = sys::fs::createTemporaryFile("hip-verify", "spv", SPVTempFile);
  if (EC) {
    result.SPIRVValidatePass = false;
    result.SPIRVValidateError = "Failed to create temp file";
    return false;
  }
  
  // Write binary to file
  {
    std::ofstream OutFile(SPVTempFile.c_str(), std::ios::binary);
    if (!OutFile) {
      sys::fs::remove(SPVTempFile);
      result.SPIRVValidatePass = false;
      result.SPIRVValidateError = "Failed to write temp file";
      return false;
    }
    OutFile.write(reinterpret_cast<const char*>(spirvBinary.data()), 
                  spirvBinary.size() * sizeof(uint32_t));
  }
  
  // Find spirv-val tool
  std::string SpirvValPath;
  auto SpirvValExe = sys::findProgramByName("spirv-val");
  if (SpirvValExe) {
    SpirvValPath = SpirvValExe.get();
  } else {
    // Try common locations
    if (sys::fs::exists("/usr/local/bin/spirv-val")) {
      SpirvValPath = "/usr/local/bin/spirv-val";
    } else if (sys::fs::exists("/usr/bin/spirv-val")) {
      SpirvValPath = "/usr/bin/spirv-val";
    } else {
      sys::fs::remove(SPVTempFile);
      result.SPIRVValidatePass = true; // Basic validation passed
      result.SPIRVValidateError = "";
      LLVM_DEBUG(dbgs() << "HipVerify: spirv-val not found, skipping full validation\n");
      return true;
    }
  }
  
  // Run spirv-val
  std::string ErrMsg;
  SmallString<256> StderrFile;
  EC = sys::fs::createTemporaryFile("hip-verify-stderr", "txt", StderrFile);
  if (EC) {
    sys::fs::remove(SPVTempFile);
    result.SPIRVValidatePass = true; // Basic validation passed
    return true;
  }
  
  SmallVector<StringRef, 4> Args{SpirvValPath, SPVTempFile};
  
  StringRef StderrFileRef(StderrFile);
  int Result = sys::ExecuteAndWait(SpirvValPath, Args, {}, {{}, {}, StderrFileRef}, 0, 0, &ErrMsg);
  
  // Read stderr output
  std::string ValidationErrors;
  if (auto BufferOrErr = MemoryBuffer::getFile(StderrFile)) {
    ValidationErrors = BufferOrErr.get()->getBuffer().str();
  }
  
  // Clean up temp files
  sys::fs::remove(SPVTempFile);
  sys::fs::remove(StderrFile);
  
  result.SPIRVValidatePass = (Result == 0);
  if (!result.SPIRVValidatePass) {
    result.SPIRVValidateError = ValidationErrors;
    // Extract first error line and remove newlines - line wrapping will handle display
    size_t newlinePos = result.SPIRVValidateError.find('\n');
    if (newlinePos != std::string::npos) {
      result.SPIRVValidateError = result.SPIRVValidateError.substr(0, newlinePos);
    }
    // Remove any remaining newlines
    std::replace(result.SPIRVValidateError.begin(), result.SPIRVValidateError.end(), '\n', ' ');
  }
  
  LLVM_DEBUG(dbgs() << "HipVerify: SPIR-V validation " << (result.SPIRVValidatePass ? "PASSED" : "FAILED") << "\n");
  return result.SPIRVValidatePass;
}

std::vector<uint32_t> HipVerifyPass::convertIRToSPIRV(Module &M, std::string &errorMsg) {
  LLVM_DEBUG(dbgs() << "HipVerify: Converting LLVM IR to SPIR-V\n");
  
  // Apply function reordering for SPIR-V compliance
  reorderFunctionsForSPIRV(M);
  
  // Create temporary files
  SmallString<256> LLTempFile;
  SmallString<256> LLStrippedTempFile;
  SmallString<256> BCTempFile;
  SmallString<256> SPVTempFile;
  
  std::error_code EC;
  EC = sys::fs::createTemporaryFile("hip-verify", "ll", LLTempFile);
  if (EC) {
    errorMsg = "Failed to create IR temp file";
    return {};
  }
  
  EC = sys::fs::createTemporaryFile("hip-verify-stripped", "ll", LLStrippedTempFile);
  if (EC) {
    sys::fs::remove(LLTempFile);
    errorMsg = "Failed to create stripped IR temp file";
    return {};
  }
  
  EC = sys::fs::createTemporaryFile("hip-verify", "bc", BCTempFile);
  if (EC) {
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    errorMsg = "Failed to create bitcode temp file";
    return {};
  }
  
  EC = sys::fs::createTemporaryFile("hip-verify", "spv", SPVTempFile);
  if (EC) {
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    errorMsg = "Failed to create SPIR-V temp file";
    return {};
  }
  
  // Write the module to IR file
  {
    std::error_code EC;
    raw_fd_ostream OS(LLTempFile, EC);
    if (EC) {
      errorMsg = "Failed to write IR file";
      sys::fs::remove(LLTempFile);
      sys::fs::remove(LLStrippedTempFile);
      sys::fs::remove(BCTempFile);
      sys::fs::remove(SPVTempFile);
      return {};
    }
    OS << M;
  }
  
  // Step 1: Strip debug information using opt
  std::string OptPath = std::string(LLVM_TOOLS_BINARY_DIR) + "/opt";
  if (!sys::fs::exists(OptPath)) {
    errorMsg = "opt not found";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }
  
  std::string ErrMsg;
  SmallVector<StringRef, 8> OptArgs{
    OptPath,
    "-strip-debug",
    LLTempFile,
    "-S",
    "-o", LLStrippedTempFile
  };
  
  // Create temporary file to capture stderr (to suppress stack traces)
  SmallString<256> OptStderrFile;
  std::error_code ECOptStderr = sys::fs::createTemporaryFile("hip-verify-opt-stderr", "txt", OptStderrFile);
  if (ECOptStderr) {
    errorMsg = "Failed to create opt stderr temp file";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }
  
  // Execute opt with stderr redirected to suppress stack traces
  StringRef OptStderrFileRef(OptStderrFile);
  int OptResult = sys::ExecuteAndWait(OptPath, OptArgs, {}, {{}, {}, OptStderrFileRef}, 0, 0, &ErrMsg);
  
  // Clean up stderr file
  sys::fs::remove(OptStderrFile);
  if (OptResult != 0) {
    errorMsg = "opt -strip-debug failed";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }
  
  // Step 2: Convert to bitcode using llvm-as
  std::string LLVMAsPath = std::string(LLVM_TOOLS_BINARY_DIR) + "/llvm-as";
  if (!sys::fs::exists(LLVMAsPath)) {
    errorMsg = "llvm-as not found";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }
  
  SmallVector<StringRef, 6> LLVMAsArgs{
    LLVMAsPath,
    LLStrippedTempFile,
    "-o", BCTempFile
  };
  
  // Create temporary file to capture stderr (to suppress potential error output)
  SmallString<256> LLVMAsStderrFile;
  std::error_code ECLLVMAsStderr = sys::fs::createTemporaryFile("hip-verify-llvm-as-stderr", "txt", LLVMAsStderrFile);
  if (ECLLVMAsStderr) {
    errorMsg = "Failed to create llvm-as stderr temp file";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }
  
  // Execute llvm-as with stderr redirected
  StringRef LLVMAsStderrFileRef(LLVMAsStderrFile);
  int LLVMAsResult = sys::ExecuteAndWait(LLVMAsPath, LLVMAsArgs, {}, {{}, {}, LLVMAsStderrFileRef}, 0, 0, &ErrMsg);
  
  // Clean up stderr file
  sys::fs::remove(LLVMAsStderrFile);
  if (LLVMAsResult != 0) {
    errorMsg = "llvm-as failed";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }
  
  // Step 3: Convert to SPIR-V using llvm-spirv
  std::string LLVMSpirvPath = std::string(LLVM_TOOLS_BINARY_DIR) + "/llvm-spirv";
  if (!sys::fs::exists(LLVMSpirvPath)) {
    errorMsg = "llvm-spirv not found";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }
  
  // DEBUG: Print which llvm-spirv we're using and its version
  // errs() << "HipVerify: Using llvm-spirv at: " << LLVMSpirvPath << "\n";
  
  // Get version of llvm-spirv
  std::string VersionErrMsg;
  SmallVector<StringRef, 2> VersionArgs{LLVMSpirvPath, "--version"};
  SmallString<256> VersionOutput;
  SmallString<256> VersionError;
  StringRef VersionOutputRef(VersionOutput);
  StringRef VersionErrorRef(VersionError);
  int VersionResult = sys::ExecuteAndWait(LLVMSpirvPath, VersionArgs, {}, 
                                         {{}, VersionOutputRef, VersionErrorRef}, 
                                         0, 0, &VersionErrMsg);
  if (VersionResult == 0) {
    // errs() << "HipVerify: llvm-spirv version output: " << VersionOutput << "\n";
  } else {
    // errs() << "HipVerify: Failed to get llvm-spirv version: " << VersionErrMsg << "\n";
  }
  
  SmallVector<StringRef, 8> Args{
    LLVMSpirvPath,
    "--spirv-max-version=1.1",
    "--spirv-ext=+all",
    "-o", SPVTempFile,
    BCTempFile
  };
  
  // DEBUG: Print the full command being executed
  // errs() << "HipVerify: Executing:";
  // for (StringRef Arg : Args) {
  //   errs() << " " << Arg;
  // }
  // errs() << "\n";
  
  // Create temporary file to capture stderr
  SmallString<256> StderrFile;
  std::error_code ECStderr = sys::fs::createTemporaryFile("hip-verify-stderr", "txt", StderrFile);
  if (ECStderr) {
    errorMsg = "Failed to create stderr temp file";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }
  
  // Execute llvm-spirv with stderr redirected to file
  StringRef StderrFileRef(StderrFile);
  int Result = sys::ExecuteAndWait(LLVMSpirvPath, Args, {}, {{}, {}, StderrFileRef}, 0, 0, &ErrMsg);
  
  // Read stderr output
  std::string StderrOutput;
  if (auto BufferOrErr = MemoryBuffer::getFile(StderrFile)) {
    StderrOutput = BufferOrErr.get()->getBuffer().str();
  }
  
  // Clean up stderr file
  sys::fs::remove(StderrFile);
  
  if (Result != 0) {
    // errs() << "HipVerify: llvm-spirv failed with code " << Result << ": " << ErrMsg << "\n";
    if (!StderrOutput.empty()) {
      // errs() << "HipVerify: llvm-spirv stderr: " << StderrOutput << "\n";
    }
    
    // Store the actual error message for the verification result
    if (!StderrOutput.empty()) {
      errorMsg = "llvm-spirv failed: " + StderrOutput;
    } else if (!ErrMsg.empty()) {
      errorMsg = "llvm-spirv failed: " + ErrMsg;
    } else {
      errorMsg = "llvm-spirv failed with code " + std::to_string(Result);
    }
    
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }
  
  // Read the SPIR-V file
  auto BufferOrErr = MemoryBuffer::getFile(SPVTempFile);
  if (!BufferOrErr) {
    errorMsg = "Failed to read SPIR-V file";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }
  
  auto Buffer = std::move(BufferOrErr.get());
  const char* Data = Buffer->getBufferStart();
  size_t Size = Buffer->getBufferSize();
  
  if (Size % sizeof(uint32_t) != 0) {
    errorMsg = "SPIR-V size not aligned";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }
  
  const uint32_t* Words = reinterpret_cast<const uint32_t*>(Data);
  size_t NumWords = Size / sizeof(uint32_t);
  
  std::vector<uint32_t> SPIRVBinary(Words, Words + NumWords);
  
  // Clean up
  sys::fs::remove(LLTempFile);
  sys::fs::remove(LLStrippedTempFile);
  sys::fs::remove(BCTempFile);
  sys::fs::remove(SPVTempFile);
  
  return SPIRVBinary;
}

void HipVerifyPass::reorderFunctionsForSPIRV(Module &M) {
  if (!isSPIRVTarget(M)) {
    return;
  }
  
  if (!needsReordering(M)) {
    return;
  }
  
  LLVM_DEBUG(dbgs() << "HipVerify: Reordering functions for SPIR-V compliance\n");
  
  // Collect functions by type
  std::vector<Function*> declarations;
  std::vector<Function*> regularDefinitions;
  std::vector<Function*> kernelDefinitions;
  
  for (Function &F : M) {
    if (F.isDeclaration()) {
      declarations.push_back(&F);
    } else if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      kernelDefinitions.push_back(&F);
    } else {
      regularDefinitions.push_back(&F);
    }
  }
  
  // Store all functions
  std::vector<Function*> allFunctions;
  for (Function &F : M) {
    allFunctions.push_back(&F);
  }
  
  // Remove all functions from module
  auto &FunctionList = M.getFunctionList();
  for (Function *F : allFunctions) {
    F->removeFromParent();
  }
  
  // Re-add in correct order: kernels, declarations, regular definitions
  for (Function *F : kernelDefinitions) {
    FunctionList.push_back(F);
  }
  for (Function *F : declarations) {
    FunctionList.push_back(F);
  }
  for (Function *F : regularDefinitions) {
    FunctionList.push_back(F);
  }
}

bool HipVerifyPass::needsReordering(Module &M) {
  bool seenDeclaration = false;
  bool seenRegularDefinition = false;
  
  for (Function &F : M) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL && !F.isDeclaration()) {
      if (seenDeclaration || seenRegularDefinition) {
        return true;
      }
    } else if (F.isDeclaration()) {
      if (seenRegularDefinition) {
        return true;
      }
      seenDeclaration = true;
    } else {
      seenRegularDefinition = true;
    }
  }
  
  return false;
}

bool HipVerifyPass::isSPIRVTarget(Module &M) {
  Triple TargetTriple(M.getTargetTriple());
  return TargetTriple.isSPIRV();
}

std::vector<std::string> HipVerifyPass::wrapText(const std::string &text, size_t width) {
  std::vector<std::string> lines;
  if (text.empty()) {
    lines.push_back("");
    return lines;
  }
  
  std::string cleanText = text;
  // Replace any remaining newlines with spaces
  std::replace(cleanText.begin(), cleanText.end(), '\n', ' ');
  
  size_t pos = 0;
  while (pos < cleanText.length()) {
    size_t end = pos + width;
    if (end >= cleanText.length()) {
      // Last chunk
      lines.push_back(cleanText.substr(pos));
      break;
    }
    
    // Find the last space before the width limit
    size_t lastSpace = cleanText.find_last_of(' ', end);
    if (lastSpace != std::string::npos && lastSpace > pos) {
      lines.push_back(cleanText.substr(pos, lastSpace - pos));
      pos = lastSpace + 1; // Skip the space
    } else {
      // No space found, break at width
      lines.push_back(cleanText.substr(pos, width));
      pos = end;
    }
  }
  
  if (lines.empty()) {
    lines.push_back("");
  }
  
  return lines;
}

void HipVerifyPass::printSummaryTable(const VerificationResult &result) {
  // Print header if this is the first result
  static bool headerPrinted = false;
  if (!headerPrinted) {
    errs() << "\n";
    errs() << "+" << std::string(20, '-') << "+" << std::string(25, '-') 
           << "+" << std::string(15, '-') << "+" << std::string(20, '-') 
           << "+" << std::string(20, '-') << "+\n";
    const char *headerModule = "Module";
    const char *headerPass = "LLVM Pass";
    const char *headerIR = "IR Validate";
    const char *headerSPIRVComp = "SPIRV Compile";
    const char *headerSPIRVVal = "SPIR-V Validate";
    errs() << format("| %-18s | %-23s | %-13s | %-18s | %-18s |\n",
                     headerModule, headerPass, headerIR, headerSPIRVComp, headerSPIRVVal);
    errs() << "+" << std::string(20, '-') << "+" << std::string(25, '-') 
           << "+" << std::string(15, '-') << "+" << std::string(20, '-') 
           << "+" << std::string(20, '-') << "+\n";
    headerPrinted = true;
  }
  
  // Truncate module name if too long
  std::string moduleName = result.ModuleName;
  if (moduleName.length() > 18) {
    moduleName = moduleName.substr(0, 15) + "...";
  }
  
  // Truncate pass name if too long
  std::string passName = result.PassName;
  if (passName.length() > 23) {
    passName = passName.substr(0, 20) + "...";
  }
  
  // Format results
  std::string irResult = result.IRValidatePass ? "PASS" : 
                         (!result.IRValidateError.empty() ? result.IRValidateError : "FAIL");
  std::string spirvCompileResult = result.SPIRVCompilePass ? "PASS" : 
                                   (!result.SPIRVCompileError.empty() ? result.SPIRVCompileError : "FAIL");
  std::string spirvValidateResult = result.SPIRVValidatePass ? "PASS" : 
                                    (!result.SPIRVValidateError.empty() ? result.SPIRVValidateError : "FAIL");
  
  // Print row
  errs() << format("| %-18s | %-23s | %-13s | %-18s | %-18s |\n",
                   moduleName.c_str(), passName.c_str(), irResult.c_str(),
                   spirvCompileResult.c_str(), spirvValidateResult.c_str());
  errs() << "+" << std::string(20, '-') << "+" << std::string(25, '-') 
         << "+" << std::string(15, '-') << "+" << std::string(20, '-') 
         << "+" << std::string(20, '-') << "+\n";
}

void HipVerifyPass::printFinalSummary() {
  if (AllResults.empty()) {
    return;
  }
  
  // Check verification mode to decide whether to print
  std::string verifyMode = HipVerifyPass::getVerificationMode();
  if (verifyMode == "off") {
    return; // Never print if disabled
  }
  
  // Filter results based on mode
  std::vector<VerificationResult> resultsToShow;
  if (verifyMode == "failures") {
    // For failures mode, only show table if Post-HIP passes fails
    // Find the Post-HIP passes result
    bool postHipPassed = true;
    for (const auto &res : AllResults) {
      if (res.PassName == "Post-HIP passes") {
        // Check if Post-HIP passes succeeded completely
        if (!res.IRValidatePass || !res.SPIRVCompilePass || !res.SPIRVValidatePass) {
          postHipPassed = false;
        }
        break;
      }
    }
    
    // Only show table if Post-HIP passes failed
    if (postHipPassed) {
      return;
    }
    
    // Show ALL results when Post-HIP passes fails (for full context and debugging)
    resultsToShow = AllResults;
  } else {
    // For "all" mode, show all results
    resultsToShow = AllResults;
  }
  
  // Get module name from first result and print it before the table
  std::string moduleName = "<unknown>";
  if (!resultsToShow.empty()) {
    moduleName = resultsToShow[0].ModuleName;
  }
  
  errs() << "\nLLVM IR and SPIR-V Validation for module: " << moduleName << "\n";
  
  // Calculate column widths based on content (no module column)
  size_t passWidth = 27;
  size_t irWidth = 30;
  size_t spirvCompileWidth = 30;
  size_t spirvValidateWidth = 60;
  
  // Print the table header
  errs() << "+" << std::string(passWidth, '-')
         << "+" << std::string(irWidth, '-') 
         << "+" << std::string(spirvCompileWidth, '-') 
         << "+" << std::string(spirvValidateWidth, '-') << "+\n";
  
  const char *headerPass = "LLVM Pass";
  const char *headerIR = "IR Validate";
  const char *headerSPIRVComp = "SPIRV Compile";
  const char *headerSPIRVVal = "SPIR-V Validate";
  
  errs() << format("| %-*s | %-*s | %-*s | %-*s |\n",
                   (int)passWidth-2, headerPass, 
                   (int)irWidth-2, headerIR,
                   (int)spirvCompileWidth-2, headerSPIRVComp, 
                   (int)spirvValidateWidth-2, headerSPIRVVal);
  
  errs() << "+" << std::string(passWidth, '-')
         << "+" << std::string(irWidth, '-') 
         << "+" << std::string(spirvCompileWidth, '-') 
         << "+" << std::string(spirvValidateWidth, '-') << "+\n";
  
  // Print each result with line wrapping
  for (const auto &result : resultsToShow) {
    // Prepare wrapped text for each column (no module column)
    std::string passName = result.PassName;
    if (passName.length() > passWidth-2) {
      passName = passName.substr(0, passWidth-5) + "...";
    }
    std::vector<std::string> passLines = {passName};
    
    // Format and wrap results
    std::string irResult = result.IRValidatePass ? "PASS" : 
                           (!result.IRValidateError.empty() ? result.IRValidateError : "FAIL");
    std::vector<std::string> irLines = HipVerifyPass::wrapText(irResult, irWidth-2);
    
    std::string spirvCompileResult = result.SPIRVCompilePass ? "PASS" : 
                                     (!result.SPIRVCompileError.empty() ? result.SPIRVCompileError : "FAIL");
    std::vector<std::string> spirvCompileLines = HipVerifyPass::wrapText(spirvCompileResult, spirvCompileWidth-2);
    
    std::string spirvValidateResult = result.SPIRVValidatePass ? "PASS" : 
                                      (!result.SPIRVValidateError.empty() ? result.SPIRVValidateError : "FAIL");
    std::vector<std::string> spirvValidateLines = HipVerifyPass::wrapText(spirvValidateResult, spirvValidateWidth-2);
    
    // Find the maximum number of lines needed
    size_t maxLines = std::max({passLines.size(), irLines.size(), 
                                spirvCompileLines.size(), spirvValidateLines.size()});
    
    // Print all lines for this result
    for (size_t lineIdx = 0; lineIdx < maxLines; ++lineIdx) {
      std::string passLine = (lineIdx < passLines.size()) ? passLines[lineIdx] : "";
      std::string irLine = (lineIdx < irLines.size()) ? irLines[lineIdx] : "";
      std::string spirvCompileLine = (lineIdx < spirvCompileLines.size()) ? spirvCompileLines[lineIdx] : "";
      std::string spirvValidateLine = (lineIdx < spirvValidateLines.size()) ? spirvValidateLines[lineIdx] : "";
      
      errs() << format("| %-*s | %-*s | %-*s | %-*s |\n",
                       (int)passWidth-2, passLine.c_str(), 
                       (int)irWidth-2, irLine.c_str(),
                       (int)spirvCompileWidth-2, spirvCompileLine.c_str(), 
                       (int)spirvValidateWidth-2, spirvValidateLine.c_str());
    }
    
    // Print separator line between results
    errs() << "+" << std::string(passWidth, '-')
           << "+" << std::string(irWidth, '-') 
           << "+" << std::string(spirvCompileWidth, '-') 
           << "+" << std::string(spirvValidateWidth, '-') << "+\n";
  }
  
  // Print summary statistics
  errs() << "\n=== HipVerify Summary ===\n";
  errs() << "Total verification points: " << AllResults.size() << "\n";
  
  int irPassed = 0, spirvCompilePassed = 0, spirvValidatePassed = 0;
  for (const auto &result : AllResults) {
    if (result.IRValidatePass) irPassed++;
    if (result.SPIRVCompilePass) spirvCompilePassed++;
    if (result.SPIRVValidatePass) spirvValidatePassed++;
  }
  
  errs() << "IR Validation: " << irPassed << "/" << AllResults.size() << " passed\n";
  errs() << "SPIR-V Compilation: " << spirvCompilePassed << "/" << AllResults.size() << " passed\n";
  errs() << "SPIR-V Validation: " << spirvValidatePassed << "/" << AllResults.size() << " passed\n";
  
  if (verifyMode == "failures" && resultsToShow.size() < AllResults.size()) {
    errs() << "Showing " << resultsToShow.size() << " entries with failures\n";
  }
  
  errs() << "==========================\n\n";
} 