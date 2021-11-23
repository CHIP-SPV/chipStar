// Removes freeze instructions from code for SPIRV-LLVM translator tool.
//
// SPIRV-LLVM translator tool does not understand freeze instructions. This file
// provides a pass to remove them as a workaround. This pass won't likely be
// needed when the SPIR-V backend lands on the LLVM.

#include "HipDefrost.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {

bool defrost(Function &F) {
  SmallPtrSet<Instruction *, 8> EraseList;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!isa<FreezeInst>(I))
        continue;
      I.replaceAllUsesWith(I.getOperand(0));
      EraseList.insert(&I);
    }
  }
  for (auto I : EraseList)
    I->eraseFromParent();
  return EraseList.size();
}

class HipDefrostLegacyPass : public FunctionPass {
public:
  static char ID;
  HipDefrostLegacyPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override { return defrost(F); }

  StringRef getPassName() const override {
    return "Remove freeze instructions.";
  }
};

} // namespace

char HipDefrostLegacyPass::ID = 0;
static RegisterPass<HipDefrostLegacyPass> X("hip-defrost",
                                            "Remove freeze instructions.");

// Pass hook for the new pass manager.
#if LLVM_VERSION_MAJOR > 11
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

PreservedAnalyses HipDefrostPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  return defrost(F) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-defrost",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-defrost") {
                    FPM.addPass(HipDefrostPass());
                    return true;
                  }
                  return false;
                });
          }};
}

#endif // LLVM_VERSION_MAJOR > 11
