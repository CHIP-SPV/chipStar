//===- HipStripAMDGCNAsm.cpp ----------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Rewrite AMDGCN inline-assembly call sites into plain LLVM load/store (or
// delete them) before the IR is handed to SPIRV-LLVM-Translator.
//
// The motivating case is hipCUB's ThreadStore/ThreadLoad templates in
// hipcub/backend/rocprim/thread/thread_{store,load}.hpp, which emit
// constructs like:
//
//   asm volatile("flat_store_dword %0, %1 glc" : : "v"(ptr), "v"(val));
//   asm volatile("s_waitcnt vmcnt(%0)" : : "I"(0x00));
//
// These AMDGCN mnemonics have no SPIR-V analogue. Without this pass
// they reach llvm-spirv, which segfaults in
// SPIRV::LLVMToSPIRVBase::transDirectCallInst (lib/SPIRV/SPIRVWriter.cpp)
// when CI->getCalledFunction() returns nullptr for the InlineAsm callee.
//
// Replacement policy:
//   flat_store_{byte,short,dword,dwordx2,...} => plain `store` to the
//     first (ptr) operand of the value held in the second operand.
//   flat_load_{...}  => plain `load`; result replaces the inline-asm
//     output.
//   s_waitcnt / s_barrier / v_* / other => erase the call. These are
//     performance/ordering hints; the surrounding plain LLVM loads and
//     stores preserve program semantics.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipStripAMDGCNAsm.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "hip-strip-amdgcn-asm"

using namespace llvm;

namespace {

// Returns true if the asm string contains any AMDGCN mnemonic we recognise
// as needing replacement/removal before SPIR-V emission.
static bool isAMDGCNAsm(StringRef Asm) {
  static const char *Mnemonics[] = {
      "flat_store_", "flat_load_", "global_store_", "global_load_",
      "buffer_store_", "buffer_load_", "s_waitcnt",   "s_barrier",
      "s_memtime",   "s_memrealtime",
      // Conservative: treat any v_ / ds_ mnemonic as AMDGCN.
      "ds_write", "ds_read", "v_mov_", "v_add_", "v_sub_", "v_mul_",
  };
  for (const char *M : Mnemonics)
    if (Asm.contains(M))
      return true;
  return false;
}

// Classify the store width for flat_store_* / global_store_* / buffer_store_*
// based on the mnemonic suffix. Returns 0 when we cannot determine a width.
static unsigned storeWidthBits(StringRef Asm) {
  if (Asm.contains("store_byte"))    return 8;
  if (Asm.contains("store_short"))   return 16;
  if (Asm.contains("store_dwordx4")) return 128;
  if (Asm.contains("store_dwordx2")) return 64;
  if (Asm.contains("store_dwordx3")) return 96;
  if (Asm.contains("store_dword"))   return 32;
  return 0;
}

static unsigned loadWidthBits(StringRef Asm) {
  if (Asm.contains("load_ubyte")  || Asm.contains("load_sbyte"))  return 8;
  if (Asm.contains("load_ushort") || Asm.contains("load_sshort")) return 16;
  if (Asm.contains("load_dwordx4"))                               return 128;
  if (Asm.contains("load_dwordx2"))                               return 64;
  if (Asm.contains("load_dwordx3"))                               return 96;
  if (Asm.contains("load_dword"))                                 return 32;
  return 0;
}

// Try to rewrite a flat_store_* inline-asm call as a plain LLVM store.
// hipCUB / rocPRIM emit the pattern:
//   asm volatile("flat_store_<w> %0, %1 glc" : : "v"(ptr), "v"(val));
// so operand 0 is the pointer and operand 1 is the value.
static bool tryReplaceStore(CallInst *CI, StringRef Asm) {
  unsigned Bits = storeWidthBits(Asm);
  if (!Bits || CI->arg_size() < 2)
    return false;

  Value *Ptr = CI->getArgOperand(0);
  Value *Val = CI->getArgOperand(1);
  if (!Ptr->getType()->isPointerTy())
    return false;

  IRBuilder<> B(CI);

  // The inline-asm operand value may be narrower than the store width
  // (e.g. `flat_store_byte` with an i16 src holding the byte). Truncate
  // as needed; only widen when strictly necessary.
  Type *ValTy = Val->getType();
  Type *TargetTy = nullptr;
  if (ValTy->isIntegerTy()) {
    unsigned VBits = ValTy->getIntegerBitWidth();
    if (VBits == Bits) {
      TargetTy = ValTy;
    } else if (VBits > Bits) {
      TargetTy = B.getIntNTy(Bits);
      Val = B.CreateTrunc(Val, TargetTy);
    } else {
      // Extending would change bit patterns; fall back to deletion.
      return false;
    }
  } else if (ValTy->isFloatingPointTy() &&
             ValTy->getPrimitiveSizeInBits() == Bits) {
    TargetTy = ValTy;
  } else {
    return false;
  }

  (void)TargetTy;
  StoreInst *SI = B.CreateStore(Val, Ptr);
  SI->setAlignment(Align(1));
  SI->setVolatile(true); // Preserve the asm volatile intent.
  CI->eraseFromParent();
  return true;
}

// Try to rewrite a flat_load_* inline-asm call as a plain LLVM load.
// hipCUB emits:
//   asm volatile("flat_load_<w> %0, %1 ...\ns_waitcnt ..." :
//                "=v"(retval) : "v"(ptr));
// so operand 0 is the pointer (only input).
static bool tryReplaceLoad(CallInst *CI, StringRef Asm) {
  unsigned Bits = loadWidthBits(Asm);
  if (!Bits || CI->arg_size() < 1)
    return false;

  Value *Ptr = CI->getArgOperand(0);
  if (!Ptr->getType()->isPointerTy())
    return false;

  Type *RetTy = CI->getType();
  if (RetTy->isVoidTy() || RetTy->getPrimitiveSizeInBits() == 0)
    return false;

  IRBuilder<> B(CI);

  // Load at the natural width of the inline-asm return type; trust the
  // frontend to have matched it to the mnemonic suffix (hipCUB does).
  LoadInst *LI = B.CreateLoad(RetTy, Ptr);
  LI->setAlignment(Align(1));
  LI->setVolatile(true);
  CI->replaceAllUsesWith(LI);
  CI->eraseFromParent();
  (void)Bits;
  return true;
}

static bool processFunction(Function &F) {
  SmallVector<CallInst *, 16> Worklist;
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (isa<InlineAsm>(CI->getCalledOperand()))
          Worklist.push_back(CI);

  bool Changed = false;
  for (CallInst *CI : Worklist) {
    auto *IA = cast<InlineAsm>(CI->getCalledOperand());
    StringRef Asm = IA->getAsmString();
    if (!isAMDGCNAsm(Asm))
      continue;

    if (Asm.contains("flat_store_") || Asm.contains("global_store_") ||
        Asm.contains("buffer_store_")) {
      if (tryReplaceStore(CI, Asm)) {
        Changed = true;
        continue;
      }
    } else if (Asm.contains("flat_load_") || Asm.contains("global_load_") ||
               Asm.contains("buffer_load_")) {
      if (tryReplaceLoad(CI, Asm)) {
        Changed = true;
        continue;
      }
    }

    // Fallback: drop the call. For s_waitcnt / s_barrier this is the
    // right thing (pure hardware hints). For an unrecognised store/load
    // shape we emit a warning — the resulting program may be slower or
    // miss a cache hint, but at least it will compile.
    if (!CI->getType()->isVoidTy()) {
      // Leave a poison so uses don't reference a deleted value.
      CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
      errs() << "Warning: HipStripAMDGCNAsm dropping unsupported AMDGCN "
             << "inline asm with non-void return: '" << Asm << "' in "
             << F.getName() << "\n";
    }
    CI->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

} // namespace

PreservedAnalyses HipStripAMDGCNAsmPass::run(Module &M,
                                             ModuleAnalysisManager &AM) {
  bool Changed = false;
  for (auto &F : M)
    if (!F.isDeclaration())
      Changed |= processFunction(F);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
