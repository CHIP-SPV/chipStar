// Define a pass plugin that runs a collection of HIP passes.

#include "HipDefrost.h"
#include "HipDynMem.h"
#include "HipTexture.h"
#include "HipStripCompilerUsed.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-passes",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-link-time-passes") {
                    // Run a collection of passes run at device link time.
                    FPM.addPass(HipStripCompilerUsedPass());
                    FPM.addPass(HipDynMemExternReplaceNewPass());
                    FPM.addPass(HipTextureExternReplaceNewPass());
                    FPM.addPass(
                        createModuleToFunctionPassAdaptor(HipDefrostPass()));
                    return true;
                  }
                  return false;
                });
          }};
}
