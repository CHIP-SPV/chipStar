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

static void addFullLinkTimePasses(ModulePassManager &MPM) {
  // Remove attributes that may prevent the device code from being optimized.
  MPM.addPass(RemoveNoInlineOptNoneAttrsPass());

  // Run a collection of passes run at device link time.
  MPM.addPass(HipStripCompilerUsedPass());
  MPM.addPass(HipDynMemExternReplaceNewPass());
  MPM.addPass(HipTextureExternReplaceNewPass());
#if LLVM_VERSION_MAJOR < 14
  // TODO: Update printf pass for HIP-Clang 14+. It now triggers an assert:
  //
  //  Assertion `isa<X>(Val) && "cast<Ty>() argument of incompatible type!"'
  //  failed.
  MPM.addPass(createModuleToFunctionPassAdaptor(HipPrintfToOpenCLPrintfPass()));
#endif
  MPM.addPass(createModuleToFunctionPassAdaptor(HipDefrostPass()));
  // This pass must appear after HipDynMemExternReplaceNewPass.
  MPM.addPass(HipGlobalVariablesPass());
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
                  return false;
                });
          }};
}
