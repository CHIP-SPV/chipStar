//===- HipStripAMDGCNAsm.h ------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM pass that strips AMDGCN-mnemonic inline assembly from the IR before
// SPIR-V emission. hipCUB's ThreadStore<STORE_CS/CG/...> templates embed
// raw AMDGCN inline asm (flat_store_dword ... glc, s_waitcnt) which
// SPIRV-LLVM-Translator cannot lower (it null-derefs in transDirectCallInst
// because CI->getCalledFunction() is nullptr for InlineAsm callees).
//
// This pass walks all CallInst whose callee is an InlineAsm, detects AMDGCN
// mnemonics in the asm string, and replaces the call with an equivalent
// plain LLVM load/store (for flat_load_*/flat_store_*) or simply deletes
// the call (for s_waitcnt/s_barrier/etc., which are pure cache/ordering
// hints with no SPIR-V analogue). Unrecognised AMDGCN mnemonics are
// deleted with a warning — correct enough for optional cache-modifier
// hints emitted by hipCUB.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_STRIP_AMDGCN_ASM_H
#define LLVM_PASSES_HIP_STRIP_AMDGCN_ASM_H

#include "llvm/IR/PassManager.h"

using namespace llvm;

class HipStripAMDGCNAsmPass : public PassInfoMixin<HipStripAMDGCNAsmPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif // LLVM_PASSES_HIP_STRIP_AMDGCN_ASM_H
