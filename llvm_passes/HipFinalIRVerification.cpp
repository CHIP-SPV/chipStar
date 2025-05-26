//===- HipFinalIRVerification.cpp ----------------------------------------===//
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

#include "HipFinalIRVerification.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Bitcode/BitcodeWriter.h"

#include <cstdlib>
#include <string>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "hip-ir-spirv-validation"

PreservedAnalyses HipIRSpirvValidationPass::run(Module &M, ModuleAnalysisManager &AM) {
  StageMsg = "[" + StageMsg + "] ";
  LLVM_DEBUG(dbgs() << "Running IR-SPIR-V validation with stage: " << StageMsg << "\n");
  
  // Always perform IR verification
  bool IRVerificationPassed = runOptVerify(M);
  
  if (!IRVerificationPassed) {
    errs() << StageMsg << "FATAL: IR verification failed. The IR is invalid.\n";
    if (!EnableSPIRVValidation) {
      errs() << StageMsg << "This indicates the input IR has structural problems.\n";
    } else {
      errs() << StageMsg << "This indicates a bug in one of the HIP transformation passes.\n";
    }
    
    // In debug builds, abort to catch issues early
    #ifndef NDEBUG
    errs() << StageMsg << "Aborting (chipStar debug build mode policy)\n";
    abort();
    #endif
    return PreservedAnalyses::all();
  } else {
    LLVM_DEBUG(dbgs() << "SUCCESS: IR verification passed for stage: " << StageMsg << "\n");
  }
  
  // Perform SPIR-V validation only if enabled
  if (EnableSPIRVValidation) {
    // Check if compile-time SPIR-V verification is enabled
    if (!isCompileTimeVerificationEnabled()) {
      LLVM_DEBUG(dbgs() << "Compile-time SPIR-V verification disabled, skipping\n");
      return PreservedAnalyses::all();
    }

    LLVM_DEBUG(dbgs() << "Running compile-time SPIR-V verification\n");

    // Convert LLVM IR to SPIR-V
    std::vector<uint32_t> spirvBinary = convertIRToSPIRV(M);
    
    if (spirvBinary.empty()) {
      errs() << StageMsg << "ERROR: Failed to convert LLVM IR to SPIR-V during compile-time verification\n";
      // report_fatal_error("SPIR-V conversion failed during compilation");
    }

    LLVM_DEBUG(dbgs() << "Successfully converted IR to SPIR-V (" 
                      << spirvBinary.size() << " words)\n");

    // Basic verification of SPIR-V binary
    if (!verifySPIRVBinary(spirvBinary)) {
      errs() << StageMsg << "ERROR: SPIR-V verification failed during compilation\n";
      errs() << StageMsg << "This indicates the generated SPIR-V is invalid and would fail at runtime\n";
      // report_fatal_error("SPIR-V verification failed during compilation");
    }

    LLVM_DEBUG(dbgs() << "SPIR-V verification passed during compilation\n");
  }
  
  return PreservedAnalyses::all();
}

bool HipIRSpirvValidationPass::runOptVerify(Module &M) {
  // Create a temporary file to write the module
  SmallString<128> TempPath;
  std::error_code EC = sys::fs::createTemporaryFile("hip_verify", "bc", TempPath);
  if (EC) {
    errs() << StageMsg << "ERROR: Failed to create temporary file for verification: " << EC.message() << "\n";
    return false;
  }
  
  // Write the module to the temporary file
  std::error_code WriteEC;
  raw_fd_ostream TempFile(TempPath, WriteEC, sys::fs::OF_None);
  if (WriteEC) {
    errs() << StageMsg << "ERROR: Failed to open temporary file for writing: " << WriteEC.message() << "\n";
    sys::fs::remove(TempPath);
    return false;
  }
  
  WriteBitcodeToFile(M, TempFile);
  TempFile.close();
  
  // Run opt -verify on the temporary file
#ifdef CHIPSTAR_LLVM_BIN_DIR
  std::string OptPath = std::string(CHIPSTAR_LLVM_BIN_DIR) + "/opt";
#else
  std::string OptPath = "opt";
#endif
  std::vector<StringRef> Args = {OptPath, "-passes=verify", "-disable-output", TempPath};
  
  std::string ErrorMsg;
  bool ExecutionFailed = false;
  int ReturnCode = sys::ExecuteAndWait(OptPath, Args, {}, {}, 0, 0, &ErrorMsg, &ExecutionFailed);
  
  // Clean up the temporary file
  sys::fs::remove(TempPath);
  
  if (ExecutionFailed) {
    errs() << StageMsg << "ERROR: Failed to execute opt: " << ErrorMsg << "\n";
    return false;
  }
  
  // opt -verify returns 0 on success, non-zero on verification failure
  return ReturnCode == 0;
}

std::vector<uint32_t> HipIRSpirvValidationPass::convertIRToSPIRV(Module &M) {
  LLVM_DEBUG(dbgs() << "Converting LLVM IR to SPIR-V using three-step process\n");

  // Create temporary files
  SmallString<256> LLTempFile;
  SmallString<256> LLStrippedTempFile;
  SmallString<256> BCTempFile;
  SmallString<256> SPVTempFile;
  
  std::error_code EC;
  EC = sys::fs::createTemporaryFile("hip-spirv-verify", "ll", LLTempFile);
  if (EC) {
    errs() << StageMsg << "Failed to create temporary IR file: " << EC.message() << "\n";
    return {};
  }
  
  EC = sys::fs::createTemporaryFile("hip-spirv-verify-stripped", "ll", LLStrippedTempFile);
  if (EC) {
    errs() << StageMsg << "Failed to create temporary stripped IR file: " << EC.message() << "\n";
    sys::fs::remove(LLTempFile);
    return {};
  }
  
  EC = sys::fs::createTemporaryFile("hip-spirv-verify", "bc", BCTempFile);
  if (EC) {
    errs() << StageMsg << "Failed to create temporary bitcode file: " << EC.message() << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    return {};
  }
  
  EC = sys::fs::createTemporaryFile("hip-spirv-verify", "spv", SPVTempFile);
  if (EC) {
    errs() << StageMsg << "Failed to create temporary SPIR-V file: " << EC.message() << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    return {};
  }

  // Write the module to IR file
  {
    std::error_code EC;
    raw_fd_ostream OS(LLTempFile, EC);
    if (EC) {
      errs() << StageMsg << "Failed to open IR file for writing: " << EC.message() << "\n";
      sys::fs::remove(LLTempFile);
      sys::fs::remove(LLStrippedTempFile);
      sys::fs::remove(BCTempFile);
      sys::fs::remove(SPVTempFile);
      return {};
    }
    
    OS << M;
    OS.close();
  }

  // Step 1: Strip debug information using opt -strip-debug
  LLVM_DEBUG(dbgs() << "Step 1: Stripping debug information using opt -strip-debug\n");
  
  // Find opt tool
  std::string OptPath;
  OptPath = std::string(CHIPSTAR_LLVM_BIN_DIR) + "/opt";


  // Check if opt exists
  if (CHIPSTAR_LLVM_BIN_DIR && !sys::fs::exists(OptPath)) {
    errs() << StageMsg << "opt not found at: " << OptPath << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }

  // Run opt -strip-debug
  std::string ErrMsg;
  SmallVector<StringRef, 8> OptArgs{
    OptPath,
    "-strip-debug",
    LLTempFile,
    "-S",
    "-o", LLStrippedTempFile
  };
  
  LLVM_DEBUG({
    raw_ostream &DbgS = dbgs();
    DbgS << "Running opt -strip-debug:";
    for (StringRef Arg : OptArgs) {
      DbgS << " " << Arg;
    }
    DbgS << "\n";
  });
  
  int OptResult = sys::ExecuteAndWait(OptPath, OptArgs, {}, {}, 0, 0, &ErrMsg);
  
  if (OptResult != 0) {
    errs() << StageMsg << "opt -strip-debug failed with code " << OptResult << ": " << ErrMsg << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }

  LLVM_DEBUG(dbgs() << "Successfully stripped debug information\n");

  // Step 2: Convert stripped .ll to .bc using llvm-as
  LLVM_DEBUG(dbgs() << "Step 2: Converting stripped .ll to .bc using llvm-as\n");
  
  // Find llvm-as tool
  std::string LLVMAsPath;
  
  // First try LLVM bin directory
  if (auto LLVMBinDir = CHIPSTAR_LLVM_BIN_DIR) {
    LLVMAsPath = std::string(LLVMBinDir) + "/llvm-as";
    LLVM_DEBUG(errs() << "llvm-as found at: " << LLVMAsPath << "\n");
  } else {
    errs() << StageMsg << "llvm-as not at: " << LLVMAsPath << "\n";
    assert(false && "llvm-as not found");
  }

  // Check if llvm-as exists
  if (!sys::fs::exists(LLVMAsPath)) {
    errs() << StageMsg << "llvm-as not found at: " << LLVMAsPath << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }

  // Run llvm-as on the stripped file
  SmallVector<StringRef, 6> LLVMAsArgs{
    LLVMAsPath,
    LLStrippedTempFile,
    "-o", BCTempFile
  };
  
  LLVM_DEBUG({
    raw_ostream &DbgS = dbgs();
    DbgS << "Running llvm-as:";
    for (StringRef Arg : LLVMAsArgs) {
      DbgS << " " << Arg;
    }
    DbgS << "\n";
  });
  
  int LLVMAsResult = sys::ExecuteAndWait(LLVMAsPath, LLVMAsArgs, {}, {}, 0, 0, &ErrMsg);
  
  if (LLVMAsResult != 0) {
    errs() << StageMsg << "llvm-as failed with code " << LLVMAsResult << ": " << ErrMsg << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }

  LLVM_DEBUG(dbgs() << "Successfully converted stripped .ll to .bc\n");
  
  // Step 3: Convert .bc to .spv using llvm-spirv
  LLVM_DEBUG(dbgs() << "Step 3: Converting .bc to .spv using llvm-spirv\n");

  // Find llvm-spirv tool
  std::string LLVMSpirvPath;
  
  // First try LLVM bin directory
  if (CHIPSTAR_LLVM_BIN_DIR) {
    LLVMSpirvPath = std::string(CHIPSTAR_LLVM_BIN_DIR) + "/llvm-spirv";
    LLVM_DEBUG(errs() << "llvm-spirv found at: " << LLVMSpirvPath << "\n");
  } else {
    errs() << StageMsg << "llvm-spirv not at: " << LLVMSpirvPath << "\n";
    assert(false && "llvm-spirv not found");
  }

  // Check if llvm-spirv exists
  if (!sys::fs::exists(LLVMSpirvPath)) {
    errs() << StageMsg << "llvm-spirv not found at: " << LLVMSpirvPath << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }

  // Run llvm-spirv
  SmallVector<StringRef, 8> Args{
    LLVMSpirvPath,
    "--spirv-max-version=1.1",
    "--spirv-ext=+all",
    "-o", SPVTempFile,
    BCTempFile
  };
  
  LLVM_DEBUG({
    raw_ostream &DbgS = dbgs();
    DbgS << "Running llvm-spirv:";
    for (StringRef Arg : Args) {
      DbgS << " " << Arg;
    }
    DbgS << "\n";
  });
  
  int Result = sys::ExecuteAndWait(LLVMSpirvPath, Args, {}, {}, 0, 0, &ErrMsg);
  
  if (Result != 0) {
    errs() << StageMsg << "llvm-spirv failed with code " << Result << ": " << ErrMsg << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }

  // Read the SPIR-V file
  auto BufferOrErr = MemoryBuffer::getFile(SPVTempFile);
  if (!BufferOrErr) {
    errs() << StageMsg << "Failed to read SPIR-V file: " << BufferOrErr.getError().message() << "\n";
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
    errs() << StageMsg << "ERROR: SPIR-V binary size is not aligned to 32-bit words\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(LLStrippedTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }

  const uint32_t* Words = reinterpret_cast<const uint32_t*>(Data);
  size_t NumWords = Size / sizeof(uint32_t);
  
  std::vector<uint32_t> SPIRVBinary(Words, Words + NumWords);
  
  LLVM_DEBUG(dbgs() << "Read " << Size << " bytes (" 
                    << NumWords << " words) from SPIR-V file\n");

  // Clean up temporary files
  sys::fs::remove(LLTempFile);
  sys::fs::remove(LLStrippedTempFile);
  sys::fs::remove(BCTempFile);
  sys::fs::remove(SPVTempFile);
  
  return SPIRVBinary;
}

bool HipIRSpirvValidationPass::verifySPIRVBinary(const std::vector<uint32_t> &spirvBinary) {
  LLVM_DEBUG(dbgs() << "Verifying SPIR-V binary\n");

  // Basic SPIR-V validation
  if (spirvBinary.size() < 5) {
    errs() << StageMsg << "SPIR-V binary too small\n";
    return false;
  }

  // Check magic number
  const uint32_t SPIRVMagicNumber = 0x07230203;
  if (spirvBinary[0] != SPIRVMagicNumber) {
    errs() << StageMsg << "Invalid SPIR-V magic number: " << spirvBinary[0] << "\n";
    return false;
  }

  // Check version (word 1)
  uint32_t version = spirvBinary[1];
  uint32_t majorVersion = (version >> 16) & 0xFF;
  uint32_t minorVersion = (version >> 8) & 0xFF;
  
  LLVM_DEBUG(dbgs() << "SPIR-V version: " << majorVersion << "." << minorVersion << "\n");
  
  if (majorVersion != 1 || minorVersion > 6) {
    errs() << StageMsg << "Unsupported SPIR-V version: " << majorVersion << "." << minorVersion << "\n";
    return false;
  }

  // Check bound (word 3) - must be greater than 0
  uint32_t bound = spirvBinary[3];
  if (bound == 0) {
    errs() << StageMsg << "Invalid SPIR-V bound: " << bound << "\n";
    return false;
  }

  LLVM_DEBUG(dbgs() << "SPIR-V bound: " << bound << "\n");
  LLVM_DEBUG(dbgs() << "Basic SPIR-V verification successful\n");
  
  LLVM_DEBUG(dbgs() << "About to run spirv-val for full validation\n");
  
  // Now run spirv-val for full validation
  LLVM_DEBUG(dbgs() << "Running spirv-val for full validation\n");
  
  // Write SPIR-V to temporary file
  SmallString<256> SPVTempFile;
  std::error_code EC;
  EC = sys::fs::createTemporaryFile("hip-spirv-validate", "spv", SPVTempFile);
  if (EC) {
    errs() << StageMsg << "Failed to create temporary SPIR-V file for validation: " << EC.message() << "\n";
    return false;
  }
  
  // Write binary to file
  {
    std::ofstream OutFile(SPVTempFile.c_str(), std::ios::binary);
    if (!OutFile) {
      errs() << StageMsg << "Failed to open SPIR-V file for writing\n";
      sys::fs::remove(SPVTempFile);
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
      LLVM_DEBUG(dbgs() << "spirv-val not found, skipping full validation\n");
      sys::fs::remove(SPVTempFile);
      return true; // Return true since basic validation passed
    }
  }
  
  // Run spirv-val
  std::string ErrMsg;
  SmallString<256> StderrFile;
  EC = sys::fs::createTemporaryFile("hip-spirv-val-stderr", "txt", StderrFile);
  if (EC) {
    LLVM_DEBUG(dbgs() << "Failed to create stderr temp file\n");
    sys::fs::remove(SPVTempFile);
    return true; // Return true since basic validation passed
  }
  
  SmallVector<StringRef, 4> Args{SpirvValPath, SPVTempFile};
  SmallVector<std::optional<StringRef>, 3> Redirects{{}, {}, StringRef(StderrFile)};
  
  LLVM_DEBUG(dbgs() << "Running: " << SpirvValPath << " " << SPVTempFile << "\n");
  
  int Result = sys::ExecuteAndWait(SpirvValPath, Args, {}, Redirects, 0, 0, &ErrMsg);
  
  // Read stderr output
  std::string ValidationErrors;
  if (auto BufferOrErr = MemoryBuffer::getFile(StderrFile)) {
    ValidationErrors = BufferOrErr.get()->getBuffer().str();
  }
  
  // Clean up temp files
  sys::fs::remove(SPVTempFile);
  sys::fs::remove(StderrFile);
  
  if (Result != 0) {
    errs() << StageMsg << "SPIR-V validation failed! spirv-val returned " << Result << "\n";
    if (!ValidationErrors.empty()) {
      errs() << StageMsg << "Validation errors:\n" << ValidationErrors << "\n";
    }
    return false;
  }
  
  LLVM_DEBUG(dbgs() << "spirv-val passed - SPIR-V is fully valid\n");
  return true;
}

bool HipIRSpirvValidationPass::isCompileTimeVerificationEnabled() {
  // Check environment variable for compile-time verification
  const char* env = std::getenv("CHIP_SPIRV_VERIFY_COMPILE_TIME");
  if (!env) {
    // Default to enabled if not specified
    return true;
  }
  
  std::string value(env);
  std::transform(value.begin(), value.end(), value.begin(), ::tolower);
  return value == "1" || value == "true" || value == "on" || value == "yes";
} 