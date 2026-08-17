//===- HipCanonicalizeGEP.cpp ---------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Rewrites single-index array-of-i8 GEPs into byte GEPs with an explicitly
// scaled index:
//
//   %g = getelementptr inbounds [4 x i8], ptr %p, i64 %i
//     ->
//   %s = shl i64 %i, 2
//   %g = getelementptr inbounds i8, ptr %p, i64 %s
//
// Both forms address the same byte, so the rewrite is semantics-preserving.
// It exists to steer the SPIR-V emitters away from an access chain form that
// the Intel Graphics Compiler miscompiles.
//
// LLVM 23's InstCombine (InstCombinerImpl::visitGetElementPtrInst, the block
// commented "Canonicalize gep %T to gep [sizeof(%T) x i8]") turns every
// single-index GEP over a non-i8 type into a GEP over [sizeof(T) x i8]. Both
// SPIR-V emitters (the in-tree backend and the SPIRV-LLVM-Translator) render
// that as an OpBitcast of the pointer to `ptr [N x uchar]` followed by an
// OpInBoundsPtrAccessChain over the array type. IGC's optimizer computes wrong
// addresses for that form: chipStar's TestSnakeMiscompileO2 produces wrong
// results on Intel GPUs, while the very same SPIR-V module gives correct
// results on Intel's CPU OpenCL device and on the GPU with -cl-opt-disable, and
// spirv-val accepts the module. The byte-GEP form emits
// OpPtrAccessChain over a uchar pointer with a byte index, which IGC handles
// correctly.
//
// Affected IGC versions: every release up to and including 2.38.2, which is
// the newest published release and what both tested distro packages ship
// (intel-igc-core-2 2.36.3 on an Arc A380, 2.38.2 on a UHD 770). IGC 2.40.0
// picked up d73553dfd19d "Skip SOA Promotion if alloca and GEP types
// mismatches", which sidesteps the miscompile by disabling SoA promotion for
// this shape, so 2.40.0 and later compute correct addresses without this
// rewrite. The indexing defect behind the guard is still present upstream;
// see intel/intel-graphics-compiler#429 and the fix proposed in #430. Drop
// this pass once the supported IGC floor is 2.40.0 or newer.
//
// Only exact matches of the canonicalization's output are rewritten: one index
// operand, an array-of-i8 source element type with a power-of-two element
// count, and an index as wide as the target's pointer index type. Narrower
// index types are skipped because a GEP sign-extends its index *after*
// scaling, so shifting first could overflow where the original could not.
//
// Constant-expression GEPs need no handling: the canonicalization above runs
// only over GetElementPtrInst, so it cannot introduce this shape into a
// ConstantExpr.
//
// Copyright (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipCanonicalizeGEP.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "PassPluginCompat.h"
#include "llvm/Support/MathExtras.h"

#define PASS_NAME "hip-canonicalize-gep"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

#if LLVM_VERSION_MAJOR >= 23

namespace {

/// Returns the element count of the array-of-i8 source element type if \p GEP
/// has exactly the shape produced by InstCombine's byte-array
/// canonicalization, and 0 otherwise.
static uint64_t getByteArrayStride(GetElementPtrInst *GEP,
                                   const DataLayout &DL) {
  if (GEP->getNumIndices() != 1)
    return 0;

  auto *ArrTy = dyn_cast<ArrayType>(GEP->getSourceElementType());
  if (!ArrTy || !ArrTy->getElementType()->isIntegerTy(8))
    return 0;

  uint64_t NumElts = ArrTy->getNumElements();
  if (NumElts < 2 || !isPowerOf2_64(NumElts))
    return 0;

  // Vector-of-pointers GEPs and vector indices are outside the shape this pass
  // recognizes.
  if (GEP->getType()->isVectorTy() || GEP->getPointerOperandType()->isVectorTy())
    return 0;

  auto *IdxTy = dyn_cast<IntegerType>(GEP->getOperand(1)->getType());
  if (!IdxTy ||
      IdxTy->getBitWidth() != DL.getIndexTypeSizeInBits(GEP->getType()))
    return 0;

  return NumElts;
}

} // namespace

PreservedAnalyses HipCanonicalizeGEPPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  const DataLayout &DL = M.getDataLayout();
  SmallVector<std::pair<GetElementPtrInst *, uint64_t>, 16> Worklist;

  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (uint64_t Stride = getByteArrayStride(GEP, DL))
            Worklist.emplace_back(GEP, Stride);

  for (auto &Entry : Worklist) {
    GetElementPtrInst *GEP = Entry.first;
    Value *Idx = GEP->getOperand(1);

    IRBuilder<> B(GEP);
    // Plain shl: the index is only widened to the pointer index type, so a
    // shift that wraps would have wrapped in the original GEP's scaling too.
    Value *Scaled = B.CreateShl(
        Idx, ConstantInt::get(Idx->getType(), Log2_64(Entry.second)),
        Idx->getName() + ".byteoff");
    Value *NewGEP =
        B.CreateGEP(B.getInt8Ty(), GEP->getPointerOperand(), Scaled,
                    GEP->getName(), GEP->getNoWrapFlags());

    GEP->replaceAllUsesWith(NewGEP);
    GEP->eraseFromParent();
  }

  return Worklist.empty() ? PreservedAnalyses::all() : PreservedAnalyses::none();
}

#else // LLVM_VERSION_MAJOR >= 23

// Older LLVM releases do not perform the array-of-i8 GEP canonicalization, so
// there is nothing to undo.
PreservedAnalyses HipCanonicalizeGEPPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  return PreservedAnalyses::all();
}

#endif // LLVM_VERSION_MAJOR >= 23

#ifndef CHIP_COMBINED_PASS_PLUGIN
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_NAME, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_NAME) {
                    MPM.addPass(HipCanonicalizeGEPPass());
                    return true;
                  }
                  return false;
                });
          }};
}
#endif // CHIP_COMBINED_PASS_PLUGIN
