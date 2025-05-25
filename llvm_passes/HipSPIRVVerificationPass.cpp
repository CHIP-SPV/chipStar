//===- HipSPIRVVerificationPass.cpp --------------------------------------===//
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

#include "HipSPIRVVerificationPass.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Bitcode/BitcodeWriter.h"

#include <cstdlib>
#include <string>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "hip-spirv-verification"

PreservedAnalyses HipSPIRVVerificationPass::run(Module &M, ModuleAnalysisManager &AM) {
  // Check if compile-time SPIR-V verification is enabled
  if (!isCompileTimeVerificationEnabled()) {
    LLVM_DEBUG(dbgs() << "Compile-time SPIR-V verification disabled, skipping\n");
    return PreservedAnalyses::all();
  }

  LLVM_DEBUG(dbgs() << "Running compile-time SPIR-V verification\n");

  // Convert LLVM IR to SPIR-V
  std::vector<uint32_t> spirvBinary = convertIRToSPIRV(M);
  
  if (spirvBinary.empty()) {
    errs() << "ERROR: Failed to convert LLVM IR to SPIR-V during compile-time verification\n";
    report_fatal_error("SPIR-V conversion failed during compilation");
  }

  LLVM_DEBUG(dbgs() << "Successfully converted IR to SPIR-V (" 
                    << spirvBinary.size() << " words)\n");

  // Basic verification of SPIR-V binary
  if (!verifySPIRVBinary(spirvBinary)) {
    errs() << "ERROR: SPIR-V verification failed during compilation\n";
    errs() << "This indicates the generated SPIR-V is invalid and would fail at runtime\n";
    report_fatal_error("SPIR-V verification failed during compilation");
  }

  LLVM_DEBUG(dbgs() << "SPIR-V verification passed during compilation\n");

  // This pass doesn't modify the IR, it only verifies the SPIR-V output
  return PreservedAnalyses::all();
}

std::vector<uint32_t> HipSPIRVVerificationPass::convertIRToSPIRV(Module &M) {
  LLVM_DEBUG(dbgs() << "Converting LLVM IR to SPIR-V using two-step process\n");

  // Create temporary files
  SmallString<256> LLTempFile;
  SmallString<256> BCTempFile;
  SmallString<256> SPVTempFile;
  
  std::error_code EC;
  EC = sys::fs::createTemporaryFile("hip-spirv-verify", "ll", LLTempFile);
  if (EC) {
    errs() << "Failed to create temporary IR file: " << EC.message() << "\n";
    return {};
  }
  
  EC = sys::fs::createTemporaryFile("hip-spirv-verify", "bc", BCTempFile);
  if (EC) {
    errs() << "Failed to create temporary bitcode file: " << EC.message() << "\n";
    sys::fs::remove(LLTempFile);
    return {};
  }
  
  EC = sys::fs::createTemporaryFile("hip-spirv-verify", "spv", SPVTempFile);
  if (EC) {
    errs() << "Failed to create temporary SPIR-V file: " << EC.message() << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(BCTempFile);
    return {};
  }

  // Write the module to IR file
  {
    std::error_code EC;
    raw_fd_ostream OS(LLTempFile, EC);
    if (EC) {
      errs() << "Failed to open IR file for writing: " << EC.message() << "\n";
      sys::fs::remove(LLTempFile);
      sys::fs::remove(BCTempFile);
      sys::fs::remove(SPVTempFile);
      return {};
    }
    
    OS << M;
    OS.close();
  }

  // Step 1: Convert .ll to .bc using llvm-as
  LLVM_DEBUG(dbgs() << "Step 1: Converting .ll to .bc using llvm-as\n");
  
  // Find llvm-as tool
  std::string LLVMAsPath;
  
  // First try LLVM bin directory
  if (auto LLVMBinDir = std::getenv("CHIPSTAR_LLVM_BIN_DIR")) {
    LLVMAsPath = std::string(LLVMBinDir) + "/llvm-as";
  } else {
    // Try to find in PATH
    auto LLVMAsExe = sys::findProgramByName("llvm-as");
    if (LLVMAsExe) {
      LLVMAsPath = LLVMAsExe.get();
    } else {
      errs() << "llvm-as not found. Make sure CHIPSTAR_LLVM_BIN_DIR is set.\n";
      sys::fs::remove(LLTempFile);
      sys::fs::remove(BCTempFile);
      sys::fs::remove(SPVTempFile);
      return {};
    }
  }

  // Check if llvm-as exists
  if (!sys::fs::exists(LLVMAsPath)) {
    errs() << "llvm-as not found at: " << LLVMAsPath << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }

  // Run llvm-as
  std::string ErrMsg;
  SmallVector<StringRef, 4> LLVMAsArgs{
    LLVMAsPath,
    LLTempFile,
    "-o", BCTempFile
  };
  
  LLVM_DEBUG({
    raw_ostream &DbgS = dbgs();
    DbgS << "Running llvm-as:";
    DbgS << " " << LLVMAsPath;
    for (StringRef Arg : LLVMAsArgs) {
      DbgS << " " << Arg;
    }
    DbgS << "\n";
  });
  
  int LLVMAsResult = sys::ExecuteAndWait(LLVMAsPath, LLVMAsArgs, {}, {}, 0, 0, &ErrMsg);
  
  if (LLVMAsResult != 0) {
    errs() << "llvm-as failed with code " << LLVMAsResult << ": " << ErrMsg << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }

  LLVM_DEBUG(dbgs() << "Successfully converted .ll to .bc\n");
  
  // Step 2: Convert .bc to .spv using llvm-spirv
  LLVM_DEBUG(dbgs() << "Step 2: Converting .bc to .spv using llvm-spirv\n");

  // Find llvm-spirv tool
  std::string LLVMSpirvPath;
  
  // First try LLVM bin directory
  if (auto LLVMBinDir = std::getenv("CHIPSTAR_LLVM_BIN_DIR")) {
    LLVMSpirvPath = std::string(LLVMBinDir) + "/llvm-spirv";
  } else {
    // Try to find in PATH
    auto LLVMSpirvExe = sys::findProgramByName("llvm-spirv");
    if (LLVMSpirvExe) {
      LLVMSpirvPath = LLVMSpirvExe.get();
    } else {
      errs() << "llvm-spirv not found. Make sure CHIPSTAR_LLVM_BIN_DIR is set.\n";
      sys::fs::remove(LLTempFile);
      sys::fs::remove(BCTempFile);
      sys::fs::remove(SPVTempFile);
      return {};
    }
  }

  // Check if llvm-spirv exists
  if (!sys::fs::exists(LLVMSpirvPath)) {
    errs() << "llvm-spirv not found at: " << LLVMSpirvPath << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }

  // Run llvm-spirv
  SmallVector<StringRef, 8> Args{
    LLVMSpirvPath,
    "--spirv-max-version=1.1",
    "--spirv-ext=+SPV_INTEL_subgroups",
    "-o", SPVTempFile,
    BCTempFile
  };
  
  LLVM_DEBUG({
    raw_ostream &DbgS = dbgs();
    DbgS << "Running llvm-spirv:";
    DbgS << " " << LLVMSpirvPath;
    for (StringRef Arg : Args) {
      DbgS << " " << Arg;
    }
    DbgS << "\n";
  });
  
  int Result = sys::ExecuteAndWait(LLVMSpirvPath, Args, {}, {}, 0, 0, &ErrMsg);
  
  if (Result != 0) {
    errs() << "llvm-spirv failed with code " << Result << ": " << ErrMsg << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }

  // Read the SPIR-V file
  auto BufferOrErr = MemoryBuffer::getFile(SPVTempFile);
  if (!BufferOrErr) {
    errs() << "Failed to read SPIR-V file: " << BufferOrErr.getError().message() << "\n";
    sys::fs::remove(LLTempFile);
    sys::fs::remove(BCTempFile);
    sys::fs::remove(SPVTempFile);
    return {};
  }

  auto Buffer = std::move(BufferOrErr.get());
  const char* Data = Buffer->getBufferStart();
  size_t Size = Buffer->getBufferSize();

  if (Size % sizeof(uint32_t) != 0) {
    errs() << "ERROR: SPIR-V binary size is not aligned to 32-bit words\n";
    sys::fs::remove(LLTempFile);
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
  sys::fs::remove(BCTempFile);
  sys::fs::remove(SPVTempFile);
  
  return SPIRVBinary;
}

bool HipSPIRVVerificationPass::verifySPIRVBinary(const std::vector<uint32_t> &spirvBinary) {
  LLVM_DEBUG(dbgs() << "Verifying SPIR-V binary\n");

  // Basic SPIR-V validation
  if (spirvBinary.size() < 5) {
    errs() << "SPIR-V binary too small\n";
    return false;
  }

  // Check magic number
  const uint32_t SPIRVMagicNumber = 0x07230203;
  if (spirvBinary[0] != SPIRVMagicNumber) {
    errs() << "Invalid SPIR-V magic number: " << spirvBinary[0] << "\n";
    return false;
  }

  // Check version (word 1)
  uint32_t version = spirvBinary[1];
  uint32_t majorVersion = (version >> 16) & 0xFF;
  uint32_t minorVersion = (version >> 8) & 0xFF;
  
  LLVM_DEBUG(dbgs() << "SPIR-V version: " << majorVersion << "." << minorVersion << "\n");
  
  if (majorVersion != 1 || minorVersion > 6) {
    errs() << "Unsupported SPIR-V version: " << majorVersion << "." << minorVersion << "\n";
    return false;
  }

  // Check bound (word 3) - must be greater than 0
  uint32_t bound = spirvBinary[3];
  if (bound == 0) {
    errs() << "Invalid SPIR-V bound: " << bound << "\n";
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
    errs() << "Failed to create temporary SPIR-V file for validation: " << EC.message() << "\n";
    return false;
  }
  
  // Write binary to file
  {
    std::ofstream OutFile(SPVTempFile.c_str(), std::ios::binary);
    if (!OutFile) {
      errs() << "Failed to open SPIR-V file for writing\n";
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
  
  int Result = sys::ExecuteAndWait(SpirvValPath, Args, {}, ArrayRef<std::optional<StringRef>>(Redirects), 0, 0, &ErrMsg);
  
  // Read stderr output
  std::string ValidationErrors;
  if (auto BufferOrErr = MemoryBuffer::getFile(StderrFile)) {
    ValidationErrors = BufferOrErr.get()->getBuffer().str();
  }
  
  // Clean up temp files
  sys::fs::remove(SPVTempFile);
  sys::fs::remove(StderrFile);
  
  if (Result != 0) {
    errs() << "SPIR-V validation failed! spirv-val returned " << Result << "\n";
    if (!ValidationErrors.empty()) {
      errs() << "Validation errors:\n" << ValidationErrors << "\n";
    }
    return false;
  }
  
  LLVM_DEBUG(dbgs() << "spirv-val passed - SPIR-V is fully valid\n");
  return true;
}

bool HipSPIRVVerificationPass::isCompileTimeVerificationEnabled() {
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