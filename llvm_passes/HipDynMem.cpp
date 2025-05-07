//===- HipDynMem.cpp ------------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM Pass to replace dynamically sized shared arrays ("extern __shared__ type[]")
// with a function argument. This is required because CUDA/HIP use a "magic variable"
// for dynamically sized shared memory, while OpenCL API uses a kernel argument
//
// (c) 2021 Paulius Velesko for Argonne National Laboratory
// (c) 2020 Michal Babej for TUNI
// (c) 2022 Michal Babej for Argonne National Laboratory
// (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//


#include "HipDynMem.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SetVector.h"
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
#include "llvm/IR/ReplaceConstant.h"

#include <iostream>
#include <set>

using namespace llvm;

#define SPIR_LOCAL_AS 3
#define GENERIC_AS 4

typedef llvm::SmallPtrSet<Function *, 16> FSet;
typedef llvm::SetVector<Function *> OrderedFSet;
typedef llvm::SmallVector<GlobalVariable *, 8> GVarVec;

class HipDynMemExternReplacePass : public ModulePass {
private:

  static bool isGVarUsedInFunction(GlobalVariable *GV, Function *F) {
    for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
      for (BasicBlock::iterator i = BB->begin(); i != BB->end(); ++i) {
        //
        // Scan through the operands of this instruction & check for GV
        //
        Instruction * I = &*i;
        for (unsigned index = 0; index < I->getNumOperands(); ++index) {
          if (GlobalVariable *ArgGV = dyn_cast<GlobalVariable>(I->getOperand(index))) {
            if (ArgGV == GV)
              return true;
          }
        }
      }
    }
    return false;
  }

  static void replaceGVarUsesWith(GlobalVariable *GV, Function *F, Value *Repl) {
    SmallVector<unsigned, 8> OperToReplace;
    for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
      for (BasicBlock::iterator i = BB->begin(); i != BB->end(); ++i) {
        //
        // Scan through the operands of this instruction & check for GV
        //
        Instruction * I = &*i;
        OperToReplace.clear();
        for (unsigned index = 0; index < I->getNumOperands(); ++index) {
          if (GlobalVariable *ArgGV = dyn_cast<GlobalVariable>(I->getOperand(index))) {
            if (ArgGV == GV)
              OperToReplace.push_back(index);
          }
        }
        for (unsigned index : OperToReplace) {
          I->setOperand(index, Repl);
        }
      }
    }
  }

  static void recursivelyFindDirectUsers(Value *V, FSet &FS) {
    for (auto *U : V->users()) {
      if (Instruction *I = dyn_cast<Instruction>(U)) {
        if (llvm::ReturnInst *RI [[maybe_unused]] = dyn_cast<ReturnInst>(U)) {
          // ignore return instructions
          continue;
        }
        Function *IF = I->getFunction();
        if (!IF)
          continue;
        FS.insert(IF);
      } else {
        recursivelyFindDirectUsers(U, FS);
      }
    }
  }

  static void recursivelyFindIndirectUsers(Value *V, OrderedFSet &FS) {
    OrderedFSet Temp;
    for (auto U : V->users()) {
      Instruction *Inst = dyn_cast<Instruction>(U);
      if (Inst) {
        Function *IF = Inst->getFunction();
        if (!IF)
          continue;
        if (FS.count(IF) == 0) {
          FS.insert(IF);
          Temp.insert(IF);
        }
      }
    }
    for (auto F : Temp) {
     recursivelyFindIndirectUsers(F, FS);
    }
  }

  static void recursivelyReplaceArrayWithPointer(Value *DestV, Value *SrcV, Type *ElemType, IRBuilder<> &B) {
    SmallVector<Instruction *> InstsToDelete;

    for (auto U : SrcV->users()) {

      if (U->getType() == nullptr)
        continue;

      if (llvm::AddrSpaceCastInst *ASCI = dyn_cast<AddrSpaceCastInst>(U)) {
        B.SetInsertPoint(ASCI);
        PointerType *PT = PointerType::get(ElemType, ASCI->getDestAddressSpace());
        Value *NewASCI = B.CreateAddrSpaceCast(DestV, PT);

        recursivelyReplaceArrayWithPointer(NewASCI, ASCI, ElemType, B);

        // check users == 0, delete old ASCI
        if (ASCI->getNumUses() == 0) {
          InstsToDelete.push_back(ASCI);
        }
        continue;
      }

      if (llvm::GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {

        B.SetInsertPoint(GEP);
        SmallVector<Value *> Indices;
        // we skip the 1st Operand (pointer) and also 2nd Operand (=first Index)
        for (unsigned i = 1; i < GEP->getNumIndices(); ++i) {
          Indices.push_back(GEP->getOperand(1+i));
        }

        Value *VV = B.CreateGEP(ElemType, DestV, Indices);
        GetElementPtrInst *NewGEP = dyn_cast<GetElementPtrInst>(VV);

        GEP->replaceAllUsesWith(NewGEP);
        if (GEP->getNumUses() == 0) {
          InstsToDelete.push_back(GEP);
        }

        continue;
      }

      if (llvm::BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {

          B.SetInsertPoint(BCI);
          Value *NewBCI = B.CreateBitCast(DestV, BCI->getDestTy());
          BCI->replaceAllUsesWith(NewBCI);

          // check users == 0, delete old BCI
          if (BCI->getNumUses() == 0) {
            InstsToDelete.push_back(BCI);
          }
        continue;
      }

      if (llvm::ReturnInst *RI = dyn_cast<ReturnInst>(U)) {
        continue;
      }

      llvm_unreachable("Unknown user type (not GEP & not AScast)");
    }

    for (auto I : InstsToDelete) {
      I->eraseFromParent();
    }

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

  static void getInstUsers(ConstantExpr *CE, SmallVector<Instruction*, 4> &Users) {
    for (Value *U: CE->users()) {
      if (Instruction *I = dyn_cast<Instruction>(U)) {
        Users.push_back(I);
      }
      if (ConstantExpr *SubCE = dyn_cast<ConstantExpr>(U)) {
        getInstUsers(SubCE, Users);
      }
    }
  }

  static void breakConstantExprs(const GVarVec &GVars) {
#if LLVM_VERSION_MAJOR < 17
    for (GlobalVariable *GV : GVars) {
      for (Value *U : GV->users()) {
        ConstantExpr *CE = dyn_cast<ConstantExpr>(U);
        if (!CE) continue;
        SmallVector<Instruction*, 4> IUsers;
        getInstUsers(CE, IUsers);
        for (Instruction *I : IUsers) {
          convertConstantExprsToInstructions(I, CE);
        }
      }
    }
#else
    SmallVector<Constant *, 8> Cnst;
    for (GlobalVariable *GV : GVars) {
      Constant *CE = cast<Constant>(GV);
      Cnst.push_back(CE);
    }
    convertUsersOfConstantsToInstructions(Cnst);
#endif
  }


  /* clones a function with an additional argument */
  static Function *cloneFunctionWithDynMemArg(Function *F, Module &M,
                                              GlobalVariable *GV) {

    SmallVector<Type *, 8> Parameters;

    // [1024 * float] (value type is not pointer)
    Type *GVTy = GV->getValueType();
    assert(dyn_cast<ArrayType>(GVTy) != nullptr);

    // float
    Type *ElemT = GVTy->getArrayElementType();

    // float addrspace(3)*
    PointerType *AS3_PTR = PointerType::get(ElemT, GV->getAddressSpace());

    for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
         i != e; ++i) {
      Parameters.push_back(i->getType());
    }
    Parameters.push_back(AS3_PTR);

    // Create the new function.
    FunctionType *FT =
        FunctionType::get(F->getReturnType(), Parameters, F->isVarArg());
    Function *NewF =
        Function::Create(FT, F->getLinkage(), F->getAddressSpace(), "", &M);
    NewF->takeName(F);
    F->setName("old_replaced_func");

    Function::arg_iterator AI = NewF->arg_begin();
    ValueToValueMapTy VV;
    for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
         i != e; ++i) {
      AI->setName(i->getName());
      VV[&*i] = &*AI;
      ++AI;
    }
    AI->setName(GV->getName() + "__hidden_dyn_local_mem");

    SmallVector<ReturnInst *, 1> RI;

#if LLVM_VERSION_MAJOR > 11
    CloneFunctionInto(NewF, F, VV, CloneFunctionChangeType::GlobalChanges, RI);
#else
    CloneFunctionInto(NewF, F, VV, true, RI);
#endif
    IRBuilder<> B(M.getContext());

    // float* (without AS, for MDNode)
    PointerType *AS0_PTR = PointerType::get(ElemT, 0);
    updateFunctionMD(NewF, M, AS0_PTR);

    // insert new function with dynamic mem = last argument
    M.getOrInsertFunction(NewF->getName(), NewF->getFunctionType(),
                          NewF->getAttributes());


    // find all calls/uses of this function...
    std::vector<CallInst *>  CallInstUses;
    for (const auto &U : F->users()) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (CI) {
        CallInstUses.push_back(CI);
      } else {
        llvm_unreachable("unknown instruction - bug");
      }
    }

    // ... and replace them with calls to new function
    for (CallInst *CI : CallInstUses) {
      llvm::SmallVector<Value *, 12> Args;
      Function *CallerF = CI->getCaller();
      assert(CallerF);
      assert(CallerF->arg_size() > 0);
      for (Value *V : CI->args()) {
        Args.push_back(V);
      }
      Argument *LastArg = CallerF->getArg(CallerF->arg_size() - 1);
      Args.push_back(LastArg);
      B.SetInsertPoint(CI);
      CallInst *NewCI = B.CreateCall(FT, NewF, Args);

      CI->replaceAllUsesWith(NewCI);
      CI->eraseFromParent();
    }

    // now we can safely delete the old function
    if(F->getNumUses() != 0)
      llvm_unreachable("old function still has uses - bug!");
    F->eraseFromParent();

    Argument *last_arg = NewF->arg_end();
    --last_arg;

    // if the function uses dynamic shared memory (via the GVar),
    // replace all uses of GVar inside function with the new dyn mem Argument
    if (isGVarUsedInFunction(GV, NewF)) {
      B.SetInsertPoint(NewF->getEntryBlock().getFirstNonPHI());

#if LLVM_VERSION_MAJOR >= 20
      // LLVM 20+ only supports opaque pointers: just replace GVar with the argument
      replaceGVarUsesWith(GV, NewF, last_arg);
#else
      if (M.getContext().supportsTypedPointers()) {
        // insert a bitcast of dyn mem argument to [N x Type] Array
        Value *BitcastV = B.CreateBitOrPointerCast(last_arg, GV->getType(), "casted_last_arg");
        Instruction *LastArgBitcast = dyn_cast<Instruction>(BitcastV);

        // replace GVar references with the [N x Type] bitcast
        replaceGVarUsesWith(GV, NewF, BitcastV);

        // replace all [N x Type]* bitcast uses with direct use of ElemT*-type dyn mem argument
        recursivelyReplaceArrayWithPointer(last_arg, LastArgBitcast, ElemT, B);

        // the bitcast to [N x Type] should now be unused
        if (LastArgBitcast->getNumUses() != 0)
          llvm_unreachable("Something still uses LastArg bitcast - bug!");
        LastArgBitcast->eraseFromParent();
      } else {
        // replace GVar references with the argument
        replaceGVarUsesWith(GV, NewF, last_arg);
      }
#endif
    }

    return NewF;
  }

  static bool transformDynamicShMemVarsImpl(Module &M) {

    bool Modified = false;

    GVarVec GVars;

    /* unfortunately the M.global_begin/end iterators hide some of the
     * global variables, therefore are not usable here; must use VST */
    ValueSymbolTable &VST = M.getValueSymbolTable();
    ValueSymbolTable::iterator VSTI;

    // find global variables that represent dynamic shared memory (__shared__)
    for (VSTI = VST.begin(); VSTI != VST.end(); ++VSTI) {

      Value *V = VSTI->getValue();
      GlobalVariable *GV = dyn_cast<GlobalVariable>(V);
      if (GV == nullptr)
        continue;

      Type *AT = nullptr;

      // Dynamic shared arrays declared as "extern __shared__ int something[]"
      // are 0 sized, and this causes problems for SPIRV translator, so we need
      // to fix them by converting to pointers
      // Dynamic shared arrays declared with HIP_DYNAMIC_SHARED macro are declared as
      // "__shared__ type var[4294967295];"
      if (GV->hasName() == true && GV->getAddressSpace() == SPIR_LOCAL_AS &&
          (AT = dyn_cast<ArrayType>(GV->getValueType())) != nullptr &&
          (AT->getArrayNumElements() == 4294967295 || AT->getArrayNumElements() == 0)
          ) {
        GVars.push_back(GV);
      }
    }

    breakConstantExprs(GVars);


    for (GlobalVariable *GV : GVars) {
      FSet DirectUserSet;

      // first, find functions that directly use the GVar. However, these may be
      // called from other functions, so we need to append the
      // dynamic shared memory argument recursively.
      recursivelyFindDirectUsers(GV, DirectUserSet);
      if (DirectUserSet.empty()) {
        continue;
      }

      OrderedFSet IndirectUserSet;
      for (Function *F : DirectUserSet) {
        recursivelyFindIndirectUsers(F, IndirectUserSet);
      }

      // find the functions that indirectly use the GVar. These will be processed (cloned with
      // dyn mem arg) before the direct users, so that the direct users
      // can rely on dyn mem argument being present in their caller.
      for (auto FI = IndirectUserSet.rbegin(); FI != IndirectUserSet.rend(); ++FI) {
        Function *F = *FI;
        Function *NewF = cloneFunctionWithDynMemArg(F, M, GV);
        if(NewF == nullptr) llvm_unreachable("cloning failed");
      }

      // now clone the direct users and replace GVar references inside them
      for (Function *F : DirectUserSet) {

        Function *NewF = cloneFunctionWithDynMemArg(F, M, GV);
        if(NewF == nullptr)
          llvm_unreachable("cloning failed");
        Modified = true;
      }

      // it seems that there are some leftover users of the GVar (ConstExprs)
      while (GV->getNumUses() > 0) {
        User *U = *GV->user_begin();
        if (Instruction *I = dyn_cast<Instruction>(U)) {
          if (I->getParent()) {
            I->eraseFromParent();
          }
        } else
        if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
          if (U->getNumUses() <= 1) {
            CE->destroyConstant();
          }
        } else
        llvm_unreachable("unknown User of Global Variable - bug!");
      }

      if (GV->getNumUses() != 0) {
        llvm_unreachable("Some uses of dynamic memory GlobalVariable still remain - bug");
      }
      GV->eraseFromParent();
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

// Identifier variable for the pass
char HipDynMemExternReplacePass::ID = 0;
static RegisterPass<HipDynMemExternReplacePass>
    X("hip-dyn-mem",
      "convert HIP dynamic shared memory to OpenCL kernel argument");


// Pass hook for the new pass manager.
#if LLVM_VERSION_MAJOR > 11
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

PreservedAnalyses HipDynMemExternReplaceNewPass::run(Module &M, ModuleAnalysisManager &AM) {
  // force the entire IR to be parsed before your pass runs
  if (auto Err = M.materializeAll())
    report_fatal_error("module materialization failed");
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
