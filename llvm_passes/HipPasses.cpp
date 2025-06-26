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
#include "HipStripUsedIntrinsics.h"
#include "HipWarps.h"
#include "HipPrintf.h"
#include "HipGlobalVariables.h"
#include "HipTextureLowering.h"
#include "HipEmitLoweredNames.h"
#include "HipKernelArgSpiller.h"
#include "HipLowerZeroLengthArrays.h"
#include "HipSanityChecks.h"
#include "HipLowerSwitch.h"
#include "HipLowerMemset.h"
#include "HipIGBADetector.h"
#include "HipPromoteInts.h"
#include "HipSpirvFunctionReorderPass.h"
#include "HipVerify.h"

#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <string>
#include <utility>

using namespace llvm;

// A predicate for internalize pass
//
// This internalizes all non-kernel functions so unused ones get removed by DCE
// pass.
static bool internalizeSPIRVFunctions(const GlobalValue &GV) {
  const auto *F = dyn_cast<Function>(&GV);
  // Returning true means preserve GV.
  return !(F && F->getCallingConv() == CallingConv::SPIR_FUNC);
}

// A pass that removes noinline and optnone attributes from functions.
class RemoveNoInlineOptNoneAttrsPass
    : public PassInfoMixin<RemoveNoInlineOptNoneAttrsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    for (auto &F : M) {
      F.removeFnAttr(Attribute::NoInline);
      F.removeFnAttr(Attribute::OptimizeNone);
    }
    return PreservedAnalyses::none();
  }
  static bool isRequired() { return true; }
};

// LLVM commit 15a1769631ff0b2b3e830b03e51ae5f54f08a0ab introduces
// 'opencl.ocl.version' module metadata into device code. This triggers an
// assertion in SPIRV-LLVM Translator if the HIP device code and linked device
// bitcode has mixed OpenCL version metadata. The commit in question inserts
// OpenCL version 0.0 in HIP compilation mode (in CUDA mode it is 2.0).
//
// This pass works around the issue until some fix is introduced in Clang. The
// issue is fixed by setting OpenCL version to the same as bitcode library
// (2.0).
class HipFixOpenCLMDPass : public PassInfoMixin<HipFixOpenCLMDPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    constexpr auto OCLVersionMDName = "opencl.ocl.version";
    if (auto *OCLVersionMD = M.getNamedMetadata(OCLVersionMDName)) {
      auto &Ctx = M.getContext();
      auto *Int32Ty = IntegerType::get(Ctx, 32);
      M.eraseNamedMetadata(OCLVersionMD);
      Metadata *OCLVerElts[] = {
          ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 2)),
          ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 0))};
      OCLVersionMD = M.getOrInsertNamedMetadata(OCLVersionMDName);
      OCLVersionMD->addOperand(MDNode::get(Ctx, OCLVerElts));
    }
    // Altering OpenCL metadata probably does not invalidate any analyses.
    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

// Insert a helper that adds a pass with HipVerify validation
template <typename PassT>
static void addPassWithVerification(ModulePassManager &MPM, PassT &&P,
                                    const std::string &Name) {
  MPM.addPass(std::forward<PassT>(P));
  // Use HipVerify pass with the name of the pass that just ran (no summary printing)
  // This will always run even if the previous pass failed
  MPM.addPass(HipVerifyPass(Name, false));
}

static void addFullLinkTimePasses(ModulePassManager &MPM) {
  MPM.addPass(HipFixOpenCLMDPass()); // must be first or else we get OCL Version mismatch
  
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

  // Run a collection of passes run at device link time.
  addPassWithVerification(MPM, HipDynMemExternReplaceNewPass(), "HipDynMemExternReplaceNewPass");
  // Should be after the HipDynMemExternReplaceNewPass which relies on detecting
  // dynamic shared memories being modeled as zero length arrays.
  addPassWithVerification(MPM, HipLowerZeroLengthArraysPass(), "HipLowerZeroLengthArraysPass");

  // Prepare device code for texture function lowering which does not yet work
  // on non-inlined code and local variables of hipTextureObject_t type.
  addPassWithVerification(MPM, RemoveNoInlineOptNoneAttrsPass(), "RemoveNoInlineOptNoneAttrsPass-2");
  // Increase getInlineParams argument for more aggressive inlining.
  addPassWithVerification(MPM, ModuleInlinerWrapperPass(getInlineParams(1000)), "ModuleInlinerWrapperPass");
#if LLVM_VERSION_MAJOR < 14
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(SROA()), "SROA");
#elif LLVM_VERSION_MAJOR < 16
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(SROAPass()), "SROAPass");
#else
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(SROAPass(SROAOptions::PreserveCFG)), "SROAPass-PreserveCFG");
#endif

  addPassWithVerification(MPM, HipTextureLoweringPass(), "HipTextureLoweringPass");

  // TODO: Update printf pass for HIP-Clang 14+. It now triggers an assert:
  //
  //  Assertion `isa<X>(Val) && "cast<Ty>() argument of incompatible type!"'
  //  failed.
  addPassWithVerification(MPM, HipPrintfToOpenCLPrintfPass(), "HipPrintfToOpenCLPrintfPass");
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipDefrostPass()), "HipDefrostPass");
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(HipLowerMemsetPass()), "HipLowerMemsetPass");
  addPassWithVerification(MPM, HipAbortPass(), "HipAbortPass");
  // This pass must appear after HipDynMemExternReplaceNewPass.
  addPassWithVerification(MPM, HipGlobalVariablesPass(), "HipGlobalVariablesPass");

  addPassWithVerification(MPM, HipWarpsPass(), "HipWarpsPass");

  // This pass must be last one that modifies kernel parameter list.
  addPassWithVerification(MPM, HipKernelArgSpillerPass(), "HipKernelArgSpillerPass");

  // Remove dead code left over by HIP lowering passes and kept alive by
  // llvm.used and llvm.compiler.used intrinsic variable.
  addPassWithVerification(MPM, HipStripUsedIntrinsicsPass(), "HipStripUsedIntrinsicsPass");

  // Internalize all __device__ functions (spir_kernels) so the follow-up DCE
  // passes cleans-ups the unused ones.
  addPassWithVerification(MPM, InternalizePass(internalizeSPIRVFunctions), "InternalizePass");
  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(DCEPass()), "DCEPass");
  addPassWithVerification(MPM, GlobalDCEPass(), "GlobalDCEPass");

  addPassWithVerification(MPM, createModuleToFunctionPassAdaptor(InferAddressSpacesPass(4)), "InferAddressSpacesPass");

  addPassWithVerification(MPM, HipIGBADetectorPass(), "HipIGBADetectorPass");

  // Fix InvalidBitWidth errors due to non-standard integer types
  addPassWithVerification(MPM, HipPromoteIntsPass(), "HipPromoteIntsPass");

  // Final verification pass with summary printing
  MPM.addPass(HipVerifyPass("Post-HIP passes", true)); // true = print final summary
}

#if LLVM_VERSION_MAJOR < 14
#define PASS_ID "hip-link-time-passes"
#else
#define PASS_ID "hip-post-link-passes"
#endif

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
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
                  // Register IR-only validation pass as standalone (legacy - use hip-verify instead)
                  if (Name == "ir-validate") {
                    MPM.addPass(HipVerifyPass("IR validation"));
                    return true;
                  }
                  // Register separate SPIR-V validation pass as standalone (legacy - use hip-verify instead)
                  if (Name == "spirv-validate") {
                    MPM.addPass(HipVerifyPass("SPIR-V validation"));
                    return true;
                  }
                  // Register merged IR+SPIR-V validation pass as standalone (legacy - use hip-verify instead)
                  if (Name == "ir-spirv-validate") {
                    MPM.addPass(HipVerifyPass("IR+SPIR-V validation"));
                    return true;
                  }
                  // Register SPIR-V function reorder pass as standalone
                  if (Name == "hip-spirv-function-reorder") {
                    MPM.addPass(HipSpirvFunctionReorderPass());
                    return true;
                  }
                  // Register unified HipVerify pass
                  if (Name == "hip-verify") {
                    MPM.addPass(HipVerifyPass());
                    return true;
                  }
                  return false;
                });
          }};
}
