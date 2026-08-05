//===- HipStripDebugInfo.cpp ----------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM pass removing debug information from HIP device code modules.
//
// Neither SPIR-V producer emits debug information our consumers accept: the
// SPIRV-LLVM-Translator emits a cyclic DebugTypeComposite reference which
// spirv-val rejects, and the in-tree SPIR-V backend emits a DebugCompilationUnit
// whose DWARF version operand is not a 32-bit unsigned OpConstant. The spec
// answer for the former, SPV_KHR_relaxed_extended_instruction, is implemented by
// none of the drivers chipStar targets.
//
// Only IGC on Intel Data Center GPU Max tolerates the malformed result, which is
// what makes gdb-oneapi usable on Aurora. Everywhere else the invalid SPIR-V
// either fails validation or makes the device compiler ICE, so the debug
// information has to go. Configure with -DCHIP_KEEP_KERNEL_DEBUG_INFO=ON to keep
// it (see llvm_passes/CMakeLists.txt); the default is to strip.
//
// Stripping here, at the head of the link-time pipeline, also means none of the
// HIP lowering passes has to keep debug metadata consistent while it erases
// globals and functions.
//===----------------------------------------------------------------------===//

#include "HipStripDebugInfo.h"

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Module.h"

using namespace llvm;

PreservedAnalyses HipStripDebugInfoPass::run(Module &M,
                                             ModuleAnalysisManager &AM) {
  return StripDebugInfo(M) ? PreservedAnalyses::none()
                           : PreservedAnalyses::all();
}
