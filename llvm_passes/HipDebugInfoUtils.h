//===- HipDebugInfoUtils.h ------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Helper for erasing a global variable without leaving its debug info behind.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_DEBUG_INFO_UTILS_H
#define LLVM_PASSES_HIP_DEBUG_INFO_UTILS_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

namespace chipstar {

/// Detach the debug info describing \p GV, which is about to be erased.
///
/// GlobalVariable::eraseFromParent() does not remove the variable's
/// DIGlobalVariableExpression from the DICompileUnit's `globals:` list. The
/// SPIR-V producers walk that list, so a stale entry is emitted as a
/// DebugGlobalVariable whose Variable operand is a DebugExpression instead of
/// an OpVariable, describing storage that no longer exists.
///
/// Consumers are not obliged to cope with that. Intel's CPU OpenCL runtime
/// segfaults on it in SPIRVToLLVMDbgTran::transGlobalVariable, taking down
/// every kernel in the module. See
/// reproducers/cpu-opencl-debug-global-variable-segfault/.
///
/// Call this immediately before erasing a global.
inline void dropGlobalDebugInfo(llvm::GlobalVariable *GV) {
  llvm::SmallVector<llvm::DIGlobalVariableExpression *, 1> GVEs;
  GV->getDebugInfo(GVEs);
  if (GVEs.empty())
    return;

  llvm::SmallPtrSet<const llvm::Metadata *, 4> Dead;
  for (auto *GVE : GVEs)
    Dead.insert(GVE);

  llvm::Module &M = *GV->getParent();
  if (auto *CUs = M.getNamedMetadata("llvm.dbg.cu")) {
    for (auto *Op : CUs->operands()) {
      auto *CU = llvm::dyn_cast<llvm::DICompileUnit>(Op);
      if (!CU)
        continue;
      llvm::SmallVector<llvm::Metadata *, 8> Keep;
      bool Changed = false;
      for (auto *G : CU->getGlobalVariables()) {
        if (Dead.count(G)) {
          Changed = true;
          continue;
        }
        Keep.push_back(G);
      }
      if (Changed)
        CU->replaceGlobalVariables(llvm::MDTuple::get(M.getContext(), Keep));
    }
  }

  GV->eraseMetadata(llvm::LLVMContext::MD_dbg);
}

} // namespace chipstar

#endif
