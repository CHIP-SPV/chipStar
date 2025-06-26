//===- HipSpirvFunctionReorderPass.h -------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SPIR-V Function Reordering Pass
//
// This pass ensures SPIR-V specification compliance by reordering functions
// in the module so that all function declarations appear before any function
// definitions. This is required by SPIR-V spec section 2.4.
//
// Additionally, this pass handles forward references in OpEntryPoint instructions
// by ensuring that kernel functions referenced by entry points are defined
// before the entry point declaration, preventing validation errors like:
// "Function declarations must appear before function definitions."
//
// The pass only activates for SPIR-V targets and only performs reordering
// when necessary (i.e., when declarations appear after definitions or when
// kernel functions might cause OpEntryPoint forward reference issues).
//
// (c) 2024 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_SPIRV_FUNCTION_REORDER_H
#define LLVM_PASSES_HIP_SPIRV_FUNCTION_REORDER_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Module.h"

using namespace llvm;

class HipSpirvFunctionReorderPass : public PassInfoMixin<HipSpirvFunctionReorderPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  /// Check if the module needs function reordering for SPIR-V compliance
  /// This includes both declaration-after-definition issues and potential
  /// OpEntryPoint forward reference problems
  bool needsReordering(Module &M);
  
  /// Reorder functions to ensure declarations appear before definitions
  /// and kernel functions are positioned to avoid OpEntryPoint forward references
  void reorderFunctionsForSPIRV(Module &M);
  
  /// Check if the target is SPIR-V
  bool isSPIRVTarget(Module &M);
};

#endif // LLVM_PASSES_HIP_SPIRV_FUNCTION_REORDER_H 