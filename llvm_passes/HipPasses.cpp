//===- HipPasses.cpp ------------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Define a pass plugin that runs a collection of HIP passes.
//
// (c) 2021 Parmance for Argonne National Laboratory and
// (c) 2022 Pekka Jääskeläinen / Intel
// (c) 2023 chipStar developers
// (c) 2024 Henry Linjamäki / Intel
//===----------------------------------------------------------------------===//

#include "HipAbort.h"
#include "HipDefrost.h"
#include "HipDynMem.h"
#include "HipStripDebugInfo.h"
#include "HipStripUsedIntrinsics.h"
#include "HipWarps.h"
#include "HipPrintf.h"
#include "HipGlobalVariables.h"
#include "HipTextureLowering.h"
#include "HipEmitLoweredNames.h"
#include "HipKernelArgSpiller.h"
#include "HipLowerZeroLengthArrays.h"
#include "HipSanityChecks.h"
#include "LLVMSPIRV.h"
#include "HipLowerSwitch.h"
#include "HipLowerHintIntrinsics.h"
#include "HipLowerFPAtomicMinMax.h"
#include "HipLowerRoundIntrinsics.h"
#include "HipLowerSubwordAtomics.h"
#include "HipLowerVolatileAccesses.h"
#include "HipIGBADetector.h"
#include "HipFunctionPointerAS.h"
#include "HipPromoteInts.h"
#include "HipLowerOverflowIntrinsics.h"
#include "HipLowerPointerVectors.h"
#include "HipVerify.h"
#include "HipCanonicalizeGEP.h"
#include "HipCoalesceDuplicatePhiPreds.h"
#include "HipFixOpenCLMD.h"
#include "HipFloatShuffle.h"
#include "HipLowerSitofpI1.h"
#include "HipOffsetBeforeCast.h"
#include "HipSharedAddrLocalInit.h"
#include "HipShuffleShift.h"
#include "RemoveNoInlineOptNoneAttrs.h"
#ifdef CHIP_LLVM_USE_INTERGRATED_SPIRV
#include "HipFPContract.h"
#include "HipLowerMemmove.h"
#endif

#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "PassPluginCompat.h"
#include "chipStarConfig.hh"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <string>
#include <utility>

using namespace llvm;

// A predicate for internalize pass. Returning true means preserve GV.
//
// Internalizes all non-kernel functions so unused ones get removed by DCE
// pass, and the Itanium vtable family (_ZTV vtable, _ZTT VTT, _ZTC construction
// vtable). Clang emits those for every polymorphic class that appears in device
// code, and an explicit instantiation gives them weak_odr linkage, which
// GlobalDCE alone must keep. A device module is self-contained, so nothing can
// link against them and the unreferenced ones can go; left in, a dead table
// whose virtual base offsets are inttoptr constants aborts llvm-spirv
// (CHIP-SPV/chipStar#1382). Also drops the per-TU __hip_cuid* markers.
static bool preserveDuringInternalize(const GlobalValue &GV) {
  if (isa<GlobalVariable>(GV)) {
    StringRef Name = GV.getName();
    return !(Name.starts_with("_ZTV") || Name.starts_with("_ZTT") ||
             Name.starts_with("_ZTC") || Name.starts_with("__hip_cuid"));
  }
  const auto *F = dyn_cast<Function>(&GV);
  return !(F && F->getCallingConv() == CallingConv::SPIR_FUNC);
}

// Insert a helper that adds a pass with HipVerify validation
template <typename PassT>
static void
addPassWithVerification(ModulePassManager &MPM, PassT &&P,
                        const std::string &Name,
                        bool Verify = HipVerifyPass::isVerificationEnabled()) {
  MPM.addPass(std::forward<PassT>(P));
  if (Verify) {
    // Use HipVerify pass with the name of the pass that just ran (no summary printing)
    // This will always run even if the previous pass failed
    MPM.addPass(HipVerifyPass(Name, false));
  }
}

static void addFullLinkTimePasses(ModulePassManager &MPM) {
  MPM.addPass(HipFixOpenCLMDPass()); // must be first or else we get OCL Version mismatch

#ifndef CHIP_KEEP_KERNEL_DEBUG_INFO
  // No SPIR-V producer emits debug information our consumers accept, so drop it
  // up front unless the build opted in (-DCHIP_KEEP_KERNEL_DEBUG_INFO=ON, which
  // only makes sense on Intel Data Center GPU Max). Doing it here also spares
  // the passes below from keeping debug metadata consistent as they erase
  // globals and functions. See HipStripDebugInfo.cpp.
  MPM.addPass(HipStripDebugInfoPass());
#endif

  // Clear any previous results at the start of a new pipeline
  HipVerifyPass::clearResults();

  // Initial verification
  MPM.addPass(HipVerifyPass("Pre-HIP passes", false)); // false = don't print summary yet

  // Use HipVerify for intermediate passes without printing summary
  addPassWithVerification(MPM, HipSanityChecksPass(), "HipSanityChecksPass");

  /// For extracting name expression to lowered name expressions (hiprtc).
  addPassWithVerification(MPM, HipEmitLoweredNamesPass(), "HipEmitLoweredNamesPass");

  // Remove attributes that may prevent the device code from being optimized.
  addPassWithVerification(MPM, RemoveNoInlineOptNoneAttrsPass(), "RemoveNoInlineOptNoneAttrsPass");

  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipLowerSwitchPass()), "HipLowerSwitchPass");

  // Before HipDynMem, which cannot rewrite a shared address in an initializer.
  addPassWithVerification(MPM, HipSharedAddrLocalInitPass(),
                          "HipSharedAddrLocalInitPass");

  // Run a collection of passes run at device link time.
  addPassWithVerification(MPM, HipDynMemExternReplaceNewPass(), "HipDynMemExternReplaceNewPass");
  // Should be after the HipDynMemExternReplaceNewPass which relies on detecting
  // dynamic shared memories being modeled as zero length arrays.
  addPassWithVerification(MPM, HipLowerZeroLengthArraysPass(), "HipLowerZeroLengthArraysPass");

  // Increase getInlineParams argument for more aggressive inlining.
  addPassWithVerification(MPM, ModuleInlinerWrapperPass(getInlineParams(1000)), "ModuleInlinerWrapperPass");
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(SROAPass(SROAOptions::PreserveCFG)), "SROAPass-PreserveCFG");

  addPassWithVerification(MPM, HipTextureLoweringPass(), "HipTextureLoweringPass");

  addPassWithVerification(MPM, HipPrintfToOpenCLPrintfPass(), "HipPrintfToOpenCLPrintfPass");
  // Before HipShuffleShift, whose index analysis does not look through freeze.
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipDefrostPass()), "HipDefrostPass");
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipLowerFPAtomicMinMaxPass()), "HipLowerFPAtomicMinMaxPass");
  // OpenCL SPIR-V consumers implement 32 and 64 bit atomics only; rewrite 8
  // and 16 bit ones onto their containing word. Runs after the fmin / fmax
  // expansion so the i16 cmpxchg it produces for half gets lowered too, and
  // before InferAddressSpaces so the word address it forms with a GEP is
  // still narrowed to the global or local address space.
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipLowerSubwordAtomicsPass()), "HipLowerSubwordAtomicsPass");
  addPassWithVerification(MPM, HipLowerRoundIntrinsicsPass(), "HipLowerRoundIntrinsicsPass");
  addPassWithVerification(MPM, HipAbortPass(), "HipAbortPass");
  // This pass must appear after HipDynMemExternReplaceNewPass.
  addPassWithVerification(MPM, HipGlobalVariablesPass(), "HipGlobalVariablesPass");
  addPassWithVerification(MPM, HipOffsetBeforeCastPass(), "HipOffsetBeforeCastPass");

  // This pass must be last one that modifies kernel parameter list.
  addPassWithVerification(MPM, HipKernelArgSpillerPass(), "HipKernelArgSpillerPass");

  // After the spiller, so the pin lands on the kernels it creates.
  addPassWithVerification(MPM, HipWarpsPass(), "HipWarpsPass");

  // After every pass that can create a copy, so a zero length one it emits is
  // erased too, and before the DCE below, so the address computations feeding
  // an erased llvm.prefetch go with it.
  addPassWithVerification(MPM, HipLowerHintIntrinsicsPass(),
                          "HipLowerHintIntrinsicsPass");

  // Remove dead code left over by HIP lowering passes and kept alive by
  // llvm.used and llvm.compiler.used intrinsic variable.
  addPassWithVerification(MPM, HipStripUsedIntrinsicsPass(), "HipStripUsedIntrinsicsPass");

  // Internalize all __device__ functions (spir_kernels) so the follow-up DCE
  // passes cleans-ups the unused ones.
  addPassWithVerification(MPM, InternalizePass(preserveDuringInternalize), "InternalizePass");
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(DCEPass()), "DCEPass");
  addPassWithVerification(MPM, GlobalDCEPass(), "GlobalDCEPass");

  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(InferAddressSpacesPass(4u)), "InferAddressSpacesPass");

  // Move vtable function pointers into the generic address space. Runs after
  // inlining and InferAddressSpaces so it only sees the indirect calls that
  // genuinely survive into the SPIR-V module.
  addPassWithVerification(MPM, HipFunctionPointerASPass(), "HipFunctionPointerASPass");

  addPassWithVerification(MPM, HipIGBADetectorPass(), "HipIGBADetectorPass");

  // A volatile global access carries CUDA's ld.volatile / st.volatile meaning
  // (an access that bypasses the core's cache) and SPIR-V's Volatile memory
  // operand does not, so rewrite them into relaxed device-scope atomics, which
  // the SPIR-V producers emit as OpAtomicLoad / OpAtomicStore.
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipLowerVolatileAccessesPass()), "HipLowerVolatileAccessesPass");

  // Fix InvalidBitWidth errors due to non-standard integer types
  addPassWithVerification(MPM, HipPromoteIntsPass(), "HipPromoteIntsPass");

  // Expand llvm.{u,s}mul.with.overflow, which clang emits for device-side
  // array-new and __builtin_mul_overflow. Left in place, the backend lane
  // emits OpUMulExtended, which IGC rejects with an undefined reference to
  // _Z20__spirv_UMulExtendedll, and the translator lane cannot lower the
  // signed intrinsic at all. Either failure takes down every kernel in the
  // module.
  addPassWithVerification(MPM, HipLowerOverflowIntrinsicsPass(),
                          "HipLowerOverflowIntrinsicsPass");

  // WORKAROUND(CHIP-SPV/chipStar#1577, llvm/llvm-project#217948): LLVM 23's
  // SROA folds a struct of pointers into <N x ptr>, which the in-tree backend
  // asserts on and llvm-spirv will not translate without
  // SPV_INTEL_masked_gather_scatter, an extension current IGC rejects. Carry
  // such values as integer vectors instead. Remove once the SPIR-V path
  // handles <N x ptr> itself.
  addPassWithVerification(MPM, HipLowerPointerVectorsPass(),
                          "HipLowerPointerVectorsPass");

#ifdef CHIP_LLVM_USE_INTERGRATED_SPIRV
  // After every pass that creates or removes FP operations or calls.
  addPassWithVerification(MPM, HipFPContractPass(), "HipFPContractPass");
  // WORKAROUND(CHIP-SPV/chipStar#1958, llvm/llvm-project (not yet reported)): see the pass.
  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(HipLowerMemmovePass()),
      "HipLowerMemmovePass");
#endif

  // Steers SPIR-V emission away from an access chain form IGC miscompiles.
  // Runs last so nothing downstream reintroduces the canonicalized shape.
  addPassWithVerification(MPM, HipCanonicalizeGEPPass(),
                          "HipCanonicalizeGEPPass");

  // WORKAROUND(CHIP-SPV/chipStar#1891, KhronosGroup/SPIRV-LLVM-Translator#3918, llvm/llvm-project#209232): see the pass.
  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(HipLowerSitofpI1Pass()),
      "HipLowerSitofpI1Pass");

  // WORKAROUND(CHIP-SPV/chipStar#1896, intel/intel-graphics-compiler (not yet reported)): see the pass.
  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(HipFloatShufflePass()),
      "HipFloatShufflePass");

#ifndef CHIP_MALI_GPU_WORKAROUNDS
  // WORKAROUND(CHIP-SPV/chipStar#1978, intel/intel-graphics-compiler: no upstream issue yet): see the pass.
  // After HipFloatShufflePass, so float shuffles shift as floats. Not on Mali,
  // whose clLinkProgram cannot link the rtdevlib.
  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(HipShuffleShiftPass()),
      "HipShuffleShiftPass");
#endif

  // WORKAROUND(CHIP-SPV/chipStar#1680, KhronosGroup/SPIRV-LLVM-Translator#3866): llvm-spirv emits one OpPhi entry per LLVM phi entry, duplicating predecessors. Remove when the pinned llvm_release branch includes #3866.
  // Last CFG change before SPIR-V emission, so nothing merges the forwarding blocks back.
  addPassWithVerification(
      MPM, createModuleToFunctionPassAdaptor(HipCoalesceDuplicatePhiPredsPass()),
      "HipCoalesceDuplicatePhiPredsPass");

  // Final verification pass with summary printing
  MPM.addPass(HipVerifyPass("Post-HIP passes", true)); // true = print final summary
}

#define PASS_ID "hip-post-link-passes"

extern "C" ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-passes", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_ID) {
                    addFullLinkTimePasses(MPM);
                    return true;
                  }
                  // Register merged IR+SPIR-V validation pass as standalone
                  if (Name == "ir-spirv-validate") {
                    MPM.addPass(HipVerifyPass("IR+SPIR-V validation"));
                    return true;
                  }
                  // Register the volatile access lowering as standalone,
                  // which makes it directly testable with opt.
                  if (Name == "hip-lower-volatile-accesses") {
                    MPM.addPass(createModuleToFunctionPassAdaptor(
                        HipLowerVolatileAccessesPass()));
                    return true;
                  }
                  return false;
                });
          }};
}
