// LLVM Pass to wrap kernels that take texture objects into new kernels that take
// OpenCL texture and sampler arguments

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "llvm/IR/TypeFinder.h"

#include <iostream>

#include <vector>
#include <set>
#include <string>

#include "HipTexture.h"

using namespace llvm;
using namespace std;

class HipTextureExternReplacePass : public ModulePass {

public:
  static char ID;
  HipTextureExternReplacePass() : ModulePass(ID) {}

  static bool transformTexWrappers(Module& M) {
    return OCLWrapperFunctions::runTexture(M);
  }
  
  bool runOnModule(Module &M) override {
    return OCLWrapperFunctions::runTexture(M);
  }

  StringRef getPassName() const override {
    return "convert HIP kernel that use texture objects into kernels that take OpenCL textures and samplers";
  }
};

char HipTextureExternReplacePass::ID = 0;
static RegisterPass<HipTextureExternReplacePass>
    X("hip-texture",
      "convert HIP kernel that use texture objects into kernels that take OpenCL textures and samplers");

// Pass hook for the new pass manager.
#if LLVM_VERSION_MAJOR > 11
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

PreservedAnalyses
HipTextureExternReplaceNewPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (HipTextureExternReplacePass::transformTexWrappers(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-texture", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-texture") {
                    FPM.addPass(HipTextureExternReplaceNewPass());
                    return true;
                  }
                  return false;
                });
          }};
}

#endif // LLVM_VERSION_MAJOR > 11
