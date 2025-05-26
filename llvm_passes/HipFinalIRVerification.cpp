//===- HipFinalIRVerification.cpp ----------------------------------------===//
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

#include "HipFinalIRVerification.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Bitcode/BitcodeWriter.h"

using namespace llvm;

#define DEBUG_TYPE "hip-ir-verification"

PreservedAnalyses HipIRVerificationPass::run(Module &M, ModuleAnalysisManager &AM) {
  const char* StageStr = getStageString();
  
  LLVM_DEBUG(dbgs() << "Running IR verification " << StageStr << " HIP passes\n");
  
  bool VerificationPassed = runOptVerify(M);
  
  if (!VerificationPassed) {
    errs() << "FATAL: IR verification failed " << StageStr << " HIP passes. The IR is invalid.\n";
    if (Stage == VerificationStage::Initial) {
      errs() << "This indicates the input IR has structural problems.\n";
    } else {
      errs() << "This indicates a bug in one of the HIP transformation passes.\n";
    }
    
    // In debug builds, abort to catch issues early
    #ifndef NDEBUG
    errs() << "Aborting (chipStar debug build mode policy)\n";
    abort();
    #endif
  } else {
    // Only print success message in debug mode to avoid spam
    LLVM_DEBUG(dbgs() << "SUCCESS: IR verification passed " << StageStr << " HIP passes\n");
  }
  
  return PreservedAnalyses::all();
}

bool HipIRVerificationPass::runOptVerify(Module &M) {
  // Create a temporary file to write the module
  SmallString<128> TempPath;
  std::error_code EC = sys::fs::createTemporaryFile("hip_verify", "bc", TempPath);
  if (EC) {
    errs() << "ERROR: Failed to create temporary file for verification: " << EC.message() << "\n";
    return false;
  }
  
  // Write the module to the temporary file
  std::error_code WriteEC;
  raw_fd_ostream TempFile(TempPath, WriteEC, sys::fs::OF_None);
  if (WriteEC) {
    errs() << "ERROR: Failed to open temporary file for writing: " << WriteEC.message() << "\n";
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
    errs() << "ERROR: Failed to execute opt: " << ErrorMsg << "\n";
    return false;
  }
  
  // opt -verify returns 0 on success, non-zero on verification failure
  return ReturnCode == 0;
} 