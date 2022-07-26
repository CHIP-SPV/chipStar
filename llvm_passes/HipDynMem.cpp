// LLVM Pass to replace dynamically sized shared arrays ("extern __shared__ type[]")
// with a function argument. This is required because CUDA/HIP use a "magic variable"
// for dynamically sized shared memory, while OpenCL API uses a kernel argument

#include "HipDynMem.h"

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

#include <iostream>
#include <set>

using namespace llvm;

#define SPIR_LOCAL_AS 3
#define GENERIC_AS 4

typedef llvm::SmallPtrSet<Function *, 8> FSet;

class HipDynMemExternReplacePass : public ModulePass {
private:
  static void recursivelyFindFunctions(Value *V, FSet &FS) {
    for (auto U : V->users()) {
      Instruction *Inst = dyn_cast<Instruction>(U);
      if (Inst) {
        Function *IF = Inst->getFunction();
        if (!IF)
          continue;
        FS.insert(IF);
      } else {
        recursivelyFindFunctions(U, FS);
      }
    }
  }

  // Recursively descend a Value's users and convert any constant expressions
  // into regular instructions. returns true if it modified Func
  static bool breakConstantExpressions(Value *Val, Function *Func) {
    bool Modified = false;
    std::vector<Value *> Users(Val->user_begin(), Val->user_end());
    for (auto *U : Users) {
      if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        // First, make sure no users of this constant expression are themselves
        // constant expressions.
        Modified |= breakConstantExpressions(U, Func);

        // Convert this constant expression to an instruction.
        llvm::Instruction *I = CE->getAsInstruction();
        I->insertBefore(&*Func->begin()->begin());
        CE->replaceAllUsesWith(I);
        CE->destroyConstant();
        Modified = true;
      }
    }
    return Modified;
  }

  static Function *getFunctionUsingGlobalVar(GlobalVariable *GV) {
    FSet FuncSet;
    recursivelyFindFunctions(GV, FuncSet);
    /* Assuming dynamic shmem variables are always used only by one function.
     * TODO is it possible there would be a local variable used by >1 func ? */
    assert(FuncSet.size() <= 1 &&
           "more than one function uses dynamic mem variable!");
    if (FuncSet.size() == 1)
      return *(FuncSet.begin());
    else
      return nullptr;
  }

  static bool isValueUsedByFunction(Value *V, Function *F) {
    FSet FuncSet;
    recursivelyFindFunctions(V, FuncSet);
    return (FuncSet.find(F) != FuncSet.end());
  }

  // get Function metadata "MDName" and append NN to it
  static void appendMD(Function *F, StringRef MDName, MDNode *NN) {
    unsigned MDKind = F->getContext().getMDKindID(MDName);
    MDNode *OldMD = F->getMetadata(MDKind);

    assert(OldMD != nullptr && OldMD->getNumOperands() > 0);

    llvm::SmallVector<llvm::Metadata *, 8> NewMDNodes;
    // copy MDnodes for original args
    for (unsigned i = 0; i < (F->arg_size() - 1); ++i) {
      Metadata *N = cast<Metadata>(OldMD->getOperand(i).get());
      assert(N != nullptr);
      NewMDNodes.push_back(N);
    }
    NewMDNodes.push_back(NN->getOperand(0).get());
    F->setMetadata(MDKind, MDNode::get(F->getContext(), NewMDNodes));
  }

  static void updateFunctionMD(Function *F, Module &M,
                               PointerType *ArgTypeWithoutAS) {
    // No need to update if the function does not have kernel metadata to begin
    // with. We update the kernel metadata because the consumer of this code may
    // get confused if the metadata is not complete (level-zero is known to
    // crash).
    if (!F->hasMetadata("kernel_arg_addr_space"))
      // Assuming that other kernel metadata kinds are absent if this one is.
      return;

    IntegerType *I32Type = IntegerType::get(M.getContext(), 32);
    MDNode *MD = MDNode::get(
        M.getContext(),
        ConstantAsMetadata::get(ConstantInt::get(I32Type, SPIR_LOCAL_AS)));
    appendMD(F, "kernel_arg_addr_space", MD);

    MD = MDNode::get(M.getContext(), MDString::get(M.getContext(), "none"));
    appendMD(F, "kernel_arg_access_qual", MD);

    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    ArgTypeWithoutAS->print(rso);
    std::string res(rso.str());

    MD = MDNode::get(M.getContext(), MDString::get(M.getContext(), res));
    appendMD(F, "kernel_arg_type", MD);
    appendMD(F, "kernel_arg_base_type", MD);

    MD = MDNode::get(M.getContext(), MDString::get(M.getContext(), ""));
    appendMD(F, "kernel_arg_type_qual", MD);
  }

  /* clones a function with an additional argument */
  static Function *cloneFunctionWithDynMemArg(Function *F, Module &M,
                                       GlobalVariable *GV) {

    SmallVector<Type *, 8> Parameters;

    // [1024 * float] AS3*
    PointerType *GVT = GV->getType();

    // AT & ELT are only for OpenCL metadata.
    // [1024 * float]
    ArrayType *AT = dyn_cast<ArrayType>(GV->getValueType());
    // float
    Type *ELT = AT->getElementType();

    for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
         i != e; ++i) {
      Parameters.push_back(i->getType());
    }
    Parameters.push_back(GVT);

    // Create the new function.
    FunctionType *FT =
        FunctionType::get(F->getReturnType(), Parameters, F->isVarArg());
    Function *NewF =
        Function::Create(FT, F->getLinkage(), F->getAddressSpace(), "", &M);
    NewF->takeName(F);

    Function::arg_iterator AI = NewF->arg_begin();
    ValueToValueMapTy VV;
    for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
         i != e; ++i) {
      AI->setName(i->getName());
      VV[&*i] = &*AI;
      ++AI;
    }
    AI->setName(Twine("__hidden_dyn_local_mem"));

    SmallVector<ReturnInst *, 1> RI;

#if LLVM_VERSION_MAJOR > 11
    CloneFunctionInto(NewF, F, VV, CloneFunctionChangeType::GlobalChanges, RI);
#else
    CloneFunctionInto(NewF, F, VV, true, RI);
#endif

    // float* (without AS, for MDNode)
    PointerType *AS0_PTR = PointerType::get(ELT, 0);
    updateFunctionMD(NewF, M, AS0_PTR);

    M.getOrInsertFunction(NewF->getName(), NewF->getFunctionType(),
                          NewF->getAttributes());
    F->eraseFromParent();

    Argument *last_arg = NewF->arg_end();
    --last_arg;

    // replace all dynamic shared mem uses with local argument
    if (GV->getNumUses() > 0) {

      IRBuilder<> B(M.getContext());
      B.SetInsertPoint(NewF->getEntryBlock().getFirstNonPHI());

      GV->replaceAllUsesWith(last_arg);
    }

    assert(GV->getNumUses() == 0 && "Some uses still remain - bug!");
    GV->eraseFromParent();

    return NewF;
  }

  static bool transformDynamicShMemVarsImpl(Module &M) {

    bool Modified = false;

    /* unfortunately the M.global_begin/end iterators hide some of the
     * global variables, therefore are not usable here; must use VST */
    ValueSymbolTable &VST = M.getValueSymbolTable();
    ValueSymbolTable::iterator VSTI;

    for (VSTI = VST.begin(); VSTI != VST.end(); ++VSTI) {

      Value *V = VSTI->getValue();
      GlobalVariable *GV = dyn_cast<GlobalVariable>(V);
      if (GV == nullptr)
        continue;

      PointerType *GVT = GV->getType();
      ArrayType *AT;
      if (GV->hasName() == true && GVT->getAddressSpace() == SPIR_LOCAL_AS &&
          (AT = dyn_cast<ArrayType>(GV->getValueType())) &&
          (AT->getArrayNumElements() == 4294967295)) {

        Function *F = getFunctionUsingGlobalVar(GV);
        if (F == nullptr) {
          continue;
        }

        breakConstantExpressions(GV, F);

        Function *NewF = cloneFunctionWithDynMemArg(F, M, GV);
        assert(NewF && "cloning failed");

        Modified = true;
      }
    }

    return Modified;
  }

public:
  static char ID;
  HipDynMemExternReplacePass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    return transformDynamicShMemVarsImpl(M);
  }

  StringRef getPassName() const override {
    return "convert HIP dynamic shared memory to OpenCL kernel argument";
  }

  static bool transformDynamicShMemVars(Module &M) {
    return transformDynamicShMemVarsImpl(M);
  }
};

char HipDynMemExternReplacePass::ID = 0;
static RegisterPass<HipDynMemExternReplacePass>
    X("hip-dyn-mem",
      "convert HIP dynamic shared memory to OpenCL kernel argument");


// Pass hook for the new pass manager.
#if LLVM_VERSION_MAJOR > 11
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

PreservedAnalyses
HipDynMemExternReplaceNewPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (HipDynMemExternReplacePass::transformDynamicShMemVars(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-dyn-mem", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-dyn-mem") {
                    FPM.addPass(HipDynMemExternReplaceNewPass());
                    return true;
                  }
                  return false;
                });
          }};
}

#endif // LLVM_VERSION_MAJOR > 11
