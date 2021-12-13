// Define a pass plugin that runs a collection of HIP passes.

#include "HipDefrost.h"
#include "HipDynMem.h"
#include "HipTexture.h"
#include "HipStripCompilerUsed.h"
#include "HipPrintf.h"
#include "HipGlobalVariables.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

static void addFullLinkTimePasses(ModulePassManager &MPM) {
  // Run a collection of passes run at device link time.
  MPM.addPass(HipStripCompilerUsedPass());
  MPM.addPass(HipDynMemExternReplaceNewPass());
  MPM.addPass(HipTextureExternReplaceNewPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(HipPrintfToOpenCLPrintfPass()));
  MPM.addPass(createModuleToFunctionPassAdaptor(HipDefrostPass()));
  // This pass must appear after HipDynMemExternReplaceNewPass.
  MPM.addPass(HipGlobalVariablesPass());
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-passes",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-link-time-passes") {
                    addFullLinkTimePasses(MPM);
                    return true;
                  }
                  return false;
                });
          }};
}
