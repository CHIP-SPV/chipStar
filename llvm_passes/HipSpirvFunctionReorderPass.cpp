//===- HipSpirvFunctionReorderPass.cpp -----------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SPIR-V Function Reordering Pass Implementation
//
// This pass ensures SPIR-V specification compliance by reordering functions
// in the module so that all function declarations appear before any function
// definitions. This addresses the common SPIR-V validation error:
// "Function declarations must appear before function definitions."
//
// Additionally, this pass handles forward references in OpEntryPoint instructions
// by ensuring that kernel functions referenced by entry points are defined
// before the entry point declaration.
//
// (c) 2024 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipSpirvFunctionReorderPass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "hip-spirv-function-reorder"

PreservedAnalyses HipSpirvFunctionReorderPass::run(Module &M, ModuleAnalysisManager &AM) {
  // Only process SPIR-V targets
  if (!isSPIRVTarget(M)) {
    return PreservedAnalyses::all();
  }
  
  // Check if reordering is needed
  if (!needsReordering(M)) {
    return PreservedAnalyses::all();
  }
  
  // Perform the reordering
  reorderFunctionsForSPIRV(M);
  
  // Function reordering changes the module structure
  return PreservedAnalyses::none();
}

bool HipSpirvFunctionReorderPass::isSPIRVTarget(Module &M) {
  Triple TargetTriple(M.getTargetTriple());
  return TargetTriple.isSPIRV();
}

bool HipSpirvFunctionReorderPass::needsReordering(Module &M) {
  bool seenDeclaration = false;
  bool seenRegularDefinition = false;
  
  // Check for proper ordering: kernels first, then declarations, then regular definitions
  for (Function &F : M) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL && !F.isDeclaration()) {
      // Kernel definitions should come first
      if (seenDeclaration || seenRegularDefinition) {
        LLVM_DEBUG(dbgs() << "Found kernel function '" << F.getName() 
                          << "' after declarations or regular definitions - reordering needed\n");
        return true;
      }
    } else if (F.isDeclaration()) {
      // Declarations should come after kernels but before regular definitions
      if (seenRegularDefinition) {
        LLVM_DEBUG(dbgs() << "Found declaration '" << F.getName() 
                          << "' after regular function definition - reordering needed\n");
        return true;
      }
      seenDeclaration = true;
    } else {
      // Regular function definitions should come last
      seenRegularDefinition = true;
    }
  }
  
  return false;
}

void HipSpirvFunctionReorderPass::reorderFunctionsForSPIRV(Module &M) {
  LLVM_DEBUG(dbgs() << "Starting function reordering for SPIR-V compliance\n");
  
  // Collect all function declarations, definitions, and kernel functions
  std::vector<Function*> declarations;
  std::vector<Function*> regularDefinitions;
  std::vector<Function*> kernelDefinitions;
  
  for (Function &F : M) {
    if (F.isDeclaration()) {
      declarations.push_back(&F);
      LLVM_DEBUG(dbgs() << "  Declaration: " << F.getName() << "\n");
    } else if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      kernelDefinitions.push_back(&F);
      LLVM_DEBUG(dbgs() << "  Kernel Definition: " << F.getName() << "\n");
    } else {
      regularDefinitions.push_back(&F);
      LLVM_DEBUG(dbgs() << "  Regular Definition: " << F.getName() << "\n");
    }
  }
  
  LLVM_DEBUG(dbgs() << "Found " << declarations.size() << " declarations, " 
                    << regularDefinitions.size() << " regular definitions, and "
                    << kernelDefinitions.size() << " kernel definitions\n");
  
  // If no reordering is needed, return early
  if (declarations.empty() && regularDefinitions.empty() && kernelDefinitions.empty()) {
    LLVM_DEBUG(dbgs() << "No functions found\n");
    return;
  }
  
  // Store all functions to reorder
  std::vector<Function*> allFunctions;
  for (Function &F : M) {
    allFunctions.push_back(&F);
  }
  
  // Remove all functions from the module temporarily
  auto &FunctionList = M.getFunctionList();
  for (Function *F : allFunctions) {
    F->removeFromParent();
  }
  
  LLVM_DEBUG(dbgs() << "Removed all functions from module\n");
  
  // Re-add functions in the correct order to avoid OpEntryPoint forward references:
  // 1. Kernel definitions FIRST (so they get the lowest IDs and are defined before OpEntryPoint)
  // 2. Function declarations 
  // 3. Regular function definitions last
  
  LLVM_DEBUG(dbgs() << "Re-adding kernel definitions first:\n");
  for (Function *F : kernelDefinitions) {
    FunctionList.push_back(F);
    LLVM_DEBUG(dbgs() << "  Added kernel definition: " << F->getName() << "\n");
  }
  
  LLVM_DEBUG(dbgs() << "Re-adding declarations:\n");
  for (Function *F : declarations) {
    FunctionList.push_back(F);
    LLVM_DEBUG(dbgs() << "  Added declaration: " << F->getName() << "\n");
  }
  
  LLVM_DEBUG(dbgs() << "Re-adding regular definitions:\n");
  for (Function *F : regularDefinitions) {
    FunctionList.push_back(F);
    LLVM_DEBUG(dbgs() << "  Added regular definition: " << F->getName() << "\n");
  }
  
  LLVM_DEBUG(dbgs() << "Function reordering completed successfully\n");
  
  // Verify the new order is correct
  bool verificationPassed = !needsReordering(M);
  LLVM_DEBUG(dbgs() << "Post-reordering verification: " 
                    << (verificationPassed ? "PASSED" : "FAILED") << "\n");
  
  if (!verificationPassed) {
    errs() << "ERROR: Function reordering failed - declarations still appear after definitions\n";
  }
} 