#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Demangle/Demangle.h"
#include <system_error>

#include <vector>
#include <string>
#include <typeinfo>

#ifndef LLVM_PASSES_CUDADV_H
#define LLVM_PASSES_CUDADV_H

using namespace llvm;

namespace {
  #define CLASS_LIST_SIZE 5 
  typedef SmallMapVector<Type* , SmallVector<Type*, CLASS_LIST_SIZE>, 32> Cls2ClsMap;
  typedef SmallVector<Type*, CLASS_LIST_SIZE> ClsList;
  typedef SmallMapVector<Type*, SmallVector<Function*, CLASS_LIST_SIZE >, 32> Cls2FuncMap;
  typedef SmallMapVector<Function*, SmallVector<Type*, 5 >, 32> Func2ClsMap;
  typedef SmallMapVector<Function*, Type*, 32> Func2SClsMap;
  typedef SmallMapVector<Value*, Type*, 32> Val2ClsMap;
  typedef std::map<StructType*, StructType*> ClassTypeMap;
  
  class VirtFuncInfo {
  public:
    VirtFuncInfo(Function* VF, StructType* Cls, int id_) : VirtFunc(VF), ClsTy(Cls), id(id_) {};
 
    // Virtual function
    Function* VirtFunc;

    // Class type
    StructType* ClsTy;

    // Object ID
    const int id;
  };

  class VTFInfo {
  public:
    VTFInfo() : VT(nullptr), VF(nullptr), offset(0), clsTy(nullptr) {};

    // Virtual table reference
    Value* VT;

    // Virtual function
    Function* VF;

    // VTable offset;
    int offset;

    // Class type
    StructType* clsTy;
  };
  
  class VTFCallSite {
  public:
    VTFCallSite() : loadVTF(nullptr), getVTFOffset(nullptr), loadVT(nullptr), VTFOffset(-1), clsTy(nullptr), VF(nullptr), WF(nullptr) {};
    
    // The load vtable function instruction
    LoadInst* loadVTF;

    // The address of virtual function 
    GetElementPtrInst* getVTFOffset;

    // The load vtable instruction
    LoadInst* loadVT;

    // The offset of virtual function
    int VTFOffset;

    // The class type
    StructType* clsTy;

    // The virtual function
    Function* VF;

    // The wrapper function
    Function* WF;
  };

  class VirtCallAnalysis;
  
  class CHAInfo {
  protected:
    // Collect class hierarchy
    void collectClsH(Module& M);

    // Check through virtual functions
    void checkVirtFuncs(Module& M);

    // Check through functions
    void checkFunctions(Module& M);

    // Check the potentially optimized offset
    void checkOptimizedOffset(GlobalVariable* VTVal, Type* clsTy);

    // Regiseter virtual function with its class and offset
    void RegisterVF(Type* clsTy, Function* VF, int offset) {
      SmallVector<Function*, CLASS_LIST_SIZE >& VFs = cls2VirtFuncs[clsTy];
      if (VFs.size() == 0)
	for (int i = 0; i < CLASS_LIST_SIZE; i ++)
	  VFs.push_back(nullptr);

      VFs[offset] = VF;
    }

    // Get the class ID
    int GetClassID(Type* ClsTy) {
      if (cls2ID.find(ClsTy) == cls2ID.end())
	return -1;
      else
	return cls2ID[ClsTy];
    }

    // Get the virtual function via class and offset
    Function* GetVirtFunc(Type* ClsTy, int offset) {
      if (cls2VirtFuncs.find(ClsTy) == cls2VirtFuncs.end())
	return nullptr;
      
      SmallVector<Function*, CLASS_LIST_SIZE >& VFs = cls2VirtFuncs[ClsTy];

      return VFs[offset];
    }
    
    // Check if there is class hierarchy relationship between two class
    bool CheckCHC(Type* clsTy1, Type* clsTy2);

    // Get the base class in a given list of class type
    Type* GetBaseClass(SmallVector<Type* , CLASS_LIST_SIZE>& ClsTys);

    // Get the base from the given class
    Type* GetBaseClass(Type* ClsTy) {
      if (cls2Bases.find(ClsTy) == cls2Bases.end())
	return nullptr;

      SmallVector<Type*, CLASS_LIST_SIZE>& bases = cls2Bases[ClsTy];
      if (bases.size() == 0)
	return nullptr;

      return bases[0];
    }
    
    // Register wrapper function for class and offset
    bool RegisterWrapper(Type* clsTy, int offset, Function* newWrapFunc);
      
    // Create wrap functions
    void createWrapperFuncs(Module& M);

    // Create wrapper function
    Function* createWrapperFunc(Function* VF, SmallVector<Type*, CLASS_LIST_SIZE>& clsTys, int offset, Module& M);
    
    // Analyze class information
    void analyze(Module& M);

    friend class VirtCallAnalysis;
    
  public:
    CHAInfo(Module& M) {
      analyze(M);
    }

    bool hasBaseCls(StructType* Ty) {
      return cls2Bases.find(Ty) != cls2Bases.end();
    }
    
    ClsList& getBaseCls(StructType* Ty) {
      return cls2Bases[Ty];
    }

    bool hasDerivedCls(StructType* Ty) {
      return cls2Deriveds.find(Ty) != cls2Deriveds.end();
    }
    
    ClsList& getDerivedCls(StructType* Ty) {
      return cls2Deriveds[Ty];
    }

    ClassTypeMap& getClassTypeMap() {
      return clsTypeMap;
    }

    // Get the original offset of the virtual function for a given class vtable reference
    int getOriginalOffset(Type* clsTy, int offset) {
      // std::map<int, int>& opt2Orig = opt2OrigOffset[clsTy];
      if (opt2OrigOffset.find(clsTy) == opt2OrigOffset.end())
	return -1;
      else
	return opt2OrigOffset[clsTy] + offset;
    }
    
    StructType* getClassType(std::string typeName) {
      std::string clsName = "class." + typeName;
      if (name2Type.find(clsName) != name2Type.end())
	return name2Type[clsName];
      else
	return nullptr;
    }

    // Get the wrapper function for the given class and virtual table offset
    Function* getWrapperFunction(Type* ClsTy, int offset) {
       SmallVector<Function*, CLASS_LIST_SIZE >& wrappers = clsVF2Wrapper[ClsTy];
       if (wrappers.size() != 0) 
	 return wrappers[offset];

       return nullptr;
    }

    // Check if given value is a reference to VTable
    bool IsVTable(Value* Val) {
      return vt2Cls.find(Val) != vt2Cls.end();
    }

    // Get the class type via argument name 
    Type* getArgClassTypeByName(Function* Func, int idx);

     // Get the class type via argument type cast
    Type* getArgClassTypeByCast(Function* Func, int idx);
    
  protected: 
    // Class to its base class 
    Cls2ClsMap cls2Bases;
    
    // Class to hts derived class
    Cls2ClsMap cls2Deriveds;

    // Class to its constructors
    Cls2FuncMap cls2Cons;

    // Class to its virtual functions
    Cls2FuncMap cls2VirtFuncs;
    
    // Virtual function to its class
    Func2ClsMap virtFunc2Cls;

    // Constructor to its class
    Func2SClsMap con2Cls;

    // VTable to its class
    Val2ClsMap vt2Cls;
    
    // The map between class type and its revised type
    ClassTypeMap clsTypeMap;

    // The class to its ID
    std::map<Type*, int> cls2ID;

    // The map between optimzied offset and original offset
    std::map<Type* , int> opt2OrigOffset;
    
    // The map between class name and class type
    std::map<std::string, StructType*> name2Type;

    // The map between virtual function and its class and vtable
    std::map<Function*, VTFInfo> VF2VTFInfo;

    // The map between class, virtual function slot and its wrap function
    Cls2FuncMap clsVF2Wrapper;
  };

  class VirtCallAnalysis {
  protected:
    // Get type name from function signature
    bool getTypeName(Function* func, int offset, std::string& typeName);
    
    // Get the vtable class reference
    StructType* checkClassRef(Value* ClsRefVal, CHAInfo& cha);

    // Check the vtable related instructions
    bool checkVTableInsts(Value* Val, CHAInfo& cha, VTFCallSite& VTFCS);
    
    // Analyze the virtual call site
    void analyzeCallSite(CallInst* callInst, Module& M, CHAInfo& cha);
    
    // Analyze the call sites
    void analyze(Module& M, CHAInfo& cha);
    
  public:
    VirtCallAnalysis(Module& M_, CHAInfo& cha_) {
      analyze(M_, cha_);
    };

    // Replace virtual function calls
    void replaceVFCalls(Module& N, CHAInfo& cha);

    // Replace the VTable setup in constructors
    void replaceVTSetups(Module& M, CHAInfo& cha);
    
    // Retrieve the virtual function info
    void retrieveVFInfos(Module& M, CHAInfo& cha);
    
  protected:
    // Virtal call and its base class
    std::map<CallInst* , VTFCallSite> VTFCSs;
  };
  
  class CUDADeVirt {
  protected:
    // Class hierarchy analysis
    void CHA(Module& M);

    // Replace type in value
    void replaceTypeInValue(Value* Val, ClassTypeMap& clsTypeMap);
    
    // Rebuild class types
    void rebuildClassTypes(Module& M, CHAInfo& cha);

    // Check the type
    bool IsStructType(Type* Ty);
    
  public:
    CUDADeVirt() {};

    void apply(Module& M);
  };

  void CHAInfo::collectClsH(Module& M) {
    int clsID = 1;
    // Collect all class types with name
    for (auto& Ty : M.getIdentifiedStructTypes()) {
      std::string typeName = Ty->getName().str();
      // typeName = typeName.substr(6);
      // llvm::outs() << " actual class name: " << typeName << "\n";
      
      name2Type[typeName] = dyn_cast<StructType>(Ty);
      cls2ID[Ty] = clsID ++;
    }

    // Collect all class type with its derived class type
    for (auto& Ty : M.getIdentifiedStructTypes()) {
      // Check class hierarchy
      if (Ty->getNumElements() > 0) {
	Type* ElemTy = Ty->getElementType(0);
	if (ElemTy->isStructTy()) {
	  std::string typeName = ((StructType* )ElemTy)->getName().str();
	  typeName = typeName.substr(6);
	  unsigned idx = typeName.find(".base");
	  if (idx != 0) {
	    typeName = typeName.substr(0, idx);
	    ElemTy = getClassType(typeName);
	  }
      
	  cls2Bases[Ty].push_back(ElemTy);
	  cls2Deriveds[ElemTy].push_back(Ty);
	}
      }
    }
  }

  void CHAInfo::checkVirtFuncs(Module& M) {
    for (llvm::GlobalVariable& global : M.globals()) {
      // llvm::outs() << "Name: " << global.getName() << "\n";

      StructType* ClsTy = nullptr;
      std::string clsName;
      // Get the class type with vtable
      const char* name_ = llvm::itaniumDemangle(global.getName().str().c_str(), nullptr, nullptr, nullptr);
      if (name_) {
	std::string ValName = name_;
	unsigned idx = ValName.find("vtable for ");
	if (idx != 0)
	  continue;

	clsName = ValName.substr(11);
	ClsTy = getClassType(clsName);
      }

      if (!ClsTy)
	continue;
      
      if (global.hasInitializer()) {
	Value* InitVal = global.getInitializer();
	if (InitVal->getType()->isStructTy()) {
	  if (ConstantStruct* CSVal = dyn_cast<ConstantStruct>(InitVal)) {
	    if (ConstantArray* CAVal = dyn_cast<ConstantArray>(CSVal->getOperand(0))) {
	      for (int i = 0; i < CAVal->getNumOperands(); i ++) {
		Value* ElemVal = CAVal->getOperand(i);
		if (Function* VF = dyn_cast<Function>(ElemVal)) {
		  VTFInfo& VTF = VF2VTFInfo[VF];
		  VTF.VF = VF;
		  VTF.VT = InitVal;
		  VTF.offset = i; 
		  VTF.clsTy = ClsTy;

		  // llvm::outs() << " class: " << clsName << " -- " << i << " --> " << VF->getName().str() << "\n";

		  // Register the map between class and virtual functions
		  RegisterVF(ClsTy, VF, VTF.offset);

		  // Register the map between virtual function and class
		  virtFunc2Cls[VF].push_back(ClsTy);

		  // Register VTable to its class
		  vt2Cls[&global] = ClsTy;
		  
		  // Check the potentially optimized offset
		  checkOptimizedOffset(&global, ClsTy);
		}
	      }
	    }
	  }
	}
	llvm::outs() << " \n";
      }
      
      llvm::outs() << "------------\n";
    }
  }

  // Check the potentially optimized offset
  void CHAInfo::checkOptimizedOffset(GlobalVariable* VTVal, Type* clsTy) {
    // Check through constructors
    SmallVector<Function*, CLASS_LIST_SIZE >& constructors = cls2Cons[clsTy];
    // llvm::outs() << "constructors: " << constructors.size() << "\n";
    for (Function* constructor : constructors) {
       for (BasicBlock& BB : * constructor) {
	for (Instruction& I : BB) {
	  if (StoreInst* storeInst = dyn_cast<StoreInst>(&I)) {
	    Value* Val = storeInst->getValueOperand();
	    // GetElementPtrInst* GEPInst = dyn_cast<GetElementPtrInst>(Val);
	    if (GEPOperator* GEPOp = dyn_cast<GEPOperator>(Val)) {
	      Value* OffsetVal = GEPOp->getOperand(GEPOp->getNumOperands() - 1);
	      if (ConstantInt* ConstVal = dyn_cast<ConstantInt>(OffsetVal)) {
		int offset = ConstVal->getValue().getSExtValue();
	        opt2OrigOffset[clsTy] = offset;
	      }
	    }
	  }
	}
      }
    }
  }

  // Check if there is class hierarchy relationship between two class
  bool CHAInfo::CheckCHC(Type* clsTy1, Type* clsTy2) {
    SmallVector<Type*, CLASS_LIST_SIZE>& bases = cls2Bases[clsTy1];
    for (int i = 0; i < bases.size(); i ++)
      if (bases[i] == clsTy2)
	return true;
    
      SmallVector<Type*, CLASS_LIST_SIZE>& deriveds = cls2Deriveds[clsTy1];
      for (int i = 0; i < deriveds.size(); i ++)
	if (deriveds[i] == clsTy2)
	  return true;
      
      return false;
  }

  // Get the base class in a given list of class type
  Type* CHAInfo::GetBaseClass(SmallVector<Type* , CLASS_LIST_SIZE>& ClsTys) {
    if (ClsTys.size() == 0)
      return nullptr;
    
    Type* CheckTy = ClsTys[0];
    bool ReCheck = true;
    while (ReCheck) { 
      ClsList& BaseClsTys = cls2Bases[CheckTy];
      if (BaseClsTys.size() > 0) {
	Type * BaseClsTy = BaseClsTys[0];
	bool NeedCheck = false;
	
	for (auto& ClsTy : ClsTys) {
	  if (ClsTy == BaseClsTy) {
	    CheckTy = ClsTy;
	    // Identify the base class was in list, then restart the check
	    NeedCheck = true;
	    
	    break;
	  }
	}
	
	if (NeedCheck)
	  // Keep checking
	  continue;
      }
      
      // No more check
      ReCheck = false;
    }
    
    return CheckTy;
  }
  
  // Check through functions
  void CHAInfo::checkFunctions(Module& M) {
    std::map<std::string, Type*> name2Cls;
    std::map<std::string, Function*> name2Func;

    // Collect class names
    for (auto& Ty : M.getIdentifiedStructTypes()) {
      std::string name = Ty->getName().str();
      name2Cls[name] = Ty;
    }

    // Collect function names
    for (auto& func : M.getFunctionList()) {
      std::string name = func.getName().str();
      // llvm::outs() << "function name: " << name << "\n";
      const char* name_ = llvm::itaniumDemangle(name.c_str(), nullptr, nullptr, nullptr); // func.getName();
      if (name_) {
	std::string funcName = name_;
	// llvm::outs() << "demangled name: " << funcName << " \n";
	name2Func[funcName] = &func;

	// Check if it is a constructor function
	unsigned idx = funcName.find("::");
	std::string clsName = funcName.substr(0, idx);
	funcName = funcName.substr(idx + 2);
	idx = funcName.find("(");
	funcName = funcName.substr(0, idx);
	if (clsName == funcName) {
	  // Register the map between class and constructors
	  cls2Cons[getClassType(clsName)].push_back(&func);
	  // Register the map between constructor and its class
	  con2Cls[&func] = getClassType(clsName);
	  // llvm::outs() << "Register " << funcName << " --> " << clsName << "\n";
	}
      }
    }
  };

  // Get the class type via argument name                                                                              
  Type* CHAInfo::getArgClassTypeByName(Function* Func, int ArgID) {
     const char* funcName_ = llvm::itaniumDemangle(Func->getName().str().c_str(), nullptr, nullptr, nullptr);   
     if (funcName_) {          
       std::string funcName = funcName_;         
       
       unsigned NumArgs = Func->arg_size();
       size_t pos = funcName.find("(");
       funcName = funcName.substr(pos + 1);

       int idx = 0;
       while (idx < NumArgs && pos != std::string::npos) {
	 pos = funcName.find(", ");
	 if (pos != std::string::npos) {
	   std::string TypeName = funcName.substr(0, pos);
	   funcName = funcName.substr(pos + 2);

	   pos = TypeName.find("*");
	   if (pos != std::string::npos) {
	     // The class type must have pointer 
	     TypeName = TypeName.substr(0, pos);
	     
	     if (idx == ArgID)
	       return getClassType(TypeName);
	   }
	   
	   idx ++;
	   
	   continue;
	 }
	 pos = funcName.find(")");
	 if (pos != std::string::npos) {
           std::string TypeName = funcName.substr(0, pos);
           if (idx == (NumArgs - 1))
             return getClassType(TypeName);

	   break;
         }
       }
     }
     
     return nullptr; 
  }

  // Get the class type via argument type cast
  Type* CHAInfo::getArgClassTypeByCast(Function* Func, int idx) {
    if (idx < 0 || idx >= Func->arg_size())
      return nullptr;

    Argument* ObjArg = Func->getArg(idx);
    // Check the use of ObjArg
    for (Value::user_iterator AUI = ObjArg->user_begin(), AUE = ObjArg->user_end(); AUI != AUE; ++ AUI) {
      if (AddrSpaceCastInst* AsInst = dyn_cast<AddrSpaceCastInst>(* AUI)) {
	for (Value::user_iterator CUI = AsInst->user_begin(), CUE = AsInst->user_end(); CUI != CUE; ++ CUI) {
	  if (GetElementPtrInst* GEPInst = dyn_cast<GetElementPtrInst>(* CUI)) {
	    Type* SrcTy = GEPInst->getSourceElementType();
	    return SrcTy;
	  }
	}
      }
    }
    
    return nullptr;
  }
  
  bool CHAInfo::RegisterWrapper(Type* clsTy, int offset, Function* newWrapFunc) {
    SmallVector<Function*, CLASS_LIST_SIZE >& wrappers = clsVF2Wrapper[clsTy];
    if (wrappers.size() != 0) {
      Function* wrapFunc = wrappers[offset];
      if (wrapFunc != nullptr) 
	// There has been a wrapper function registered
	return false;
    } else {
      // Allocate wrapper space
      for (int i = 0; i < CLASS_LIST_SIZE; i ++)
	wrappers.push_back(nullptr);
    }

    if (newWrapFunc != nullptr) 
      // Register the wrapper function as it is
      wrappers[offset] = newWrapFunc;

    return true;
  }
  
  void CHAInfo::createWrapperFuncs(Module& M) { 
    for (auto& pair : VF2VTFInfo) {
      Function* VF = pair.first;
      VTFInfo& VTF = pair.second;
      Type* clsTy = VTF.clsTy;

      // Check if the wrapper function has been created or not
      if (!RegisterWrapper(clsTy, VTF.offset, nullptr))
	continue;

      // Collect class hierarchy information
      SmallVector<Type*, CLASS_LIST_SIZE> ClsTys;
      ClsTys.push_back(clsTy);
      if (Type* BaseClsTy = GetBaseClass(clsTy))
	ClsTys.push_back(BaseClsTy);
      
      // Check the VTable list again
      for (auto& checkPair : VF2VTFInfo) {
	Function* NextVF = checkPair.first;
	if (VF == NextVF)
	  // Same virtual function and class
	  continue;

	VTFInfo& NextVTF = checkPair.second;
	
	if (VTF.offset != NextVTF.offset)
	  // Not same virtual function slot
	  continue;

	// Check if current class has been collected
	bool HasCollected = false;
	for (auto& CheckTy : ClsTys) {
	  if (CheckTy == NextVTF.clsTy) {
	    HasCollected = true;
	    break;
	  } 
	}

	if (HasCollected)
	  continue;
	
	for (auto& CheckTy : ClsTys) {
	  if (CheckCHC(CheckTy, NextVTF.clsTy)) 
	    ClsTys.push_back(NextVTF.clsTy);
	}
      }
      
      // Create the wrapper function
      Function* wrapperFunc = createWrapperFunc(VF, ClsTys, VTF.offset, M);

      // Register wrapper function
      for (auto& checkTy : ClsTys)
	RegisterWrapper(checkTy, VTF.offset, wrapperFunc);
    }
  }

  // Create wrapper function
  Function* CHAInfo::createWrapperFunc(Function* VF, SmallVector<Type*, CLASS_LIST_SIZE>& ClsTys,
				       int Offset, Module& M) {
    if (VF == nullptr || ClsTys.size() == 0)
      return nullptr;

    // M.dump();

    // The the base class
    Type* BaseClsTy = GetBaseClass(ClsTys);

    // Extract the function type from virtual function
    FunctionType* VFTy = VF->getFunctionType();
    Type* RetTy = VFTy->getReturnType();
    std::vector<Type* > VFParamTys = VFTy->params().vec();
    
    FunctionType* FuncTy = FunctionType::get(RetTy, VFParamTys, false);
    Function* WrapFunc = Function::Create(FuncTy, Function::ExternalLinkage, "wrapper", M);
    
    // Create function body
    BasicBlock* EntryBB = BasicBlock::Create(M.getContext(), "entry", WrapFunc);
    IRBuilder<> Builder(EntryBB);

    // xxx ceate dummy return check 1
    /*Value* RetVal_ = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);                                         
    Builder.CreateRet(RetVal_);
    if (RetVal_)
      return WrapFunc;
    */
    
    Argument* ObjArg = WrapFunc->getArg(0);    
    // Load the object ID
    Type* IDTy = Type::getInt32Ty(M.getContext());
    Value* GEPInst = Builder.CreateGEP(BaseClsTy, ObjArg, {ConstantInt::get(IDTy, 0), ConstantInt::get(IDTy, 0)}, // {0, 0},
				       "gep_obj_id");
    Value* LoadObjIDInst = Builder.CreateLoad(IDTy, GEPInst, "load_objid");

    // Set predecessor and current basic block
    BasicBlock* PredBB = EntryBB;
    BasicBlock* CurrBB = nullptr;
    
    SmallVector<Value*, 8> Args;
    for (int i = 0; i < WrapFunc->arg_size(); i ++)
      Args.push_back(WrapFunc->getArg(i));
    
    int idx = 0;
    for (; idx < ClsTys.size() - 1; idx ++) {
      // Create current basic block
      CurrBB = BasicBlock::Create(M.getContext(), "virt_func", WrapFunc);
      // Create successor basic block
      BasicBlock* SuccBB = BasicBlock::Create(M.getContext(), "virt_func", WrapFunc);

      IRBuilder<> PredBuilder(PredBB);
      IRBuilder<> CurrBuilder(CurrBB);

      // Create branch instruction
      Constant* ConstObjID = ConstantInt::get(IDTy, GetClassID(ClsTys[idx]));
      Value* CmpInst = PredBuilder.CreateICmpNE(LoadObjIDInst, ConstObjID, "cmp_obj_id");

      // Create branch instruction
      PredBuilder.CreateCondBr(CmpInst, SuccBB, CurrBB);
      
      // Add invocation of function
      Value* CallInst = CurrBuilder.CreateCall(GetVirtFunc(ClsTys[idx], Offset), Args);

      if (!RetTy->isVoidTy()) {
	// Add the return if it is needed
	Value* RetInst = CurrBuilder.CreateRet(CallInst);
      }
      
      // Reset the predecessor
      PredBB = SuccBB; 
    }

    // VirtFuncInfo& VFInfo = virtFuncInfos[idx];

    // Create current basic block
    CurrBB = PredBB;
      
    // Add invocation of function
    IRBuilder<> CurrBuilder(CurrBB);
    
    // Add invocation of function
    Value* CallInst = CurrBuilder.CreateCall(GetVirtFunc(ClsTys[idx], Offset), Args);

    if (!RetTy->isVoidTy()) {
      // Add the return if it is needed
      Value* RetInst = CurrBuilder.CreateRet(CallInst);
    }
    
    WrapFunc->dump();
    
    return WrapFunc;
  }
  
  void CHAInfo::analyze(Module& M) {
    // Collect class hierarchy
    collectClsH(M);

    // Check through functions
    checkFunctions(M);
    
    // Check through virtual functions
    checkVirtFuncs(M);

    // Synthesize the wrap function
    createWrapperFuncs(M);
  }
  
  bool CUDADeVirt::IsStructType(Type* Ty) {
     if (isa<StructType>(Ty))
       return true;
     else if (Ty->isOpaquePointerTy() && Ty->getNumContainedTypes()) {
       Type* PtrTy = Ty->getPointerElementType();
       if (isa<StructType>(Ty->getPointerElementType()))
	 return true;
     } else if (Ty->isPointerTy() && Ty->getNumContainedTypes()) {
     
     }

     return false;
  }

  bool VirtCallAnalysis::getTypeName(Function* func, int offset, std::string& clsTypeName) {
    // Get the argument type from function signature, i.e. demangle the function name
    const char* funcName_ = llvm::itaniumDemangle(func->getName().str().c_str(), nullptr, nullptr, nullptr);
    if (funcName_) {
      std::string funcName = funcName_;
      size_t pos = funcName.find("(");
      funcName = funcName.substr(pos + 1);
      std::string typeName = "no_type";
      int count = 0;
      do {
	pos = funcName.find(", ");
	if (pos != std::string::npos) {
	  typeName = funcName.substr(0, pos - 1);
	  funcName = funcName.substr(pos + 3);
	} else {
	  pos = funcName.find(")");
	  if (pos != std::string::npos) {
	    typeName = funcName.substr(0, pos - 1);
	  }
	}
	count ++;
      } while (count < offset);

      if ((count - 1) == offset) {
	clsTypeName = typeName;
     	return true;
      }
    }

    return false;
  }
  
  StructType* VirtCallAnalysis::checkClassRef(Value* ClsRefVal, CHAInfo& cha) {
     Type* ClsRefTy = ClsRefVal->getType();
     
     if (ClsRefTy->isPointerTy()) {
       if (ClsRefTy->getNumContainedTypes()) {
	 ClsRefTy = ClsRefTy->getPointerElementType();
	 if (StructType* resTy = dyn_cast<StructType>(ClsRefTy))
	   return resTy;
       }
       
       // Trace the definition
       if (LoadInst* loadInst = dyn_cast<LoadInst>(ClsRefVal)) {
	 if (AllocaInst* allocInst = dyn_cast<AllocaInst>(loadInst->getPointerOperand())) {
	   for (Value::use_iterator ui = allocInst->use_begin(), ue = allocInst->use_end(); ui != ue; ++ ui) { 
	     if (StoreInst* storeInst = dyn_cast<StoreInst>(ui->getUser())) {
	       if (Argument* arg = dyn_cast<Argument>(storeInst->getValueOperand())) {
		 Function* func = loadInst->getParent()->getParent();
		 // Get the id of the function argument
		 int offset = -1; unsigned numArgs = func->arg_size();
		 for (int i = 0; i < (int)func->arg_size(); i ++) {
		   if (arg == func->getArg(i)) {
		     offset = i;
		     break;
		   }
		 }

		 if (offset >= 0) {
		   std::string typeName;
		   if (getTypeName(func, offset, typeName)) {
		     return cha.getClassType(typeName);
		   }
		 }
	       }
	     }
	   }
	 }
       }
     }
     
     return nullptr;;
  }

  // Check VTable instruction and fill in the virtual call related information
  bool VirtCallAnalysis::checkVTableInsts(Value* Val, CHAInfo& cha, VTFCallSite& VTFCS) {
    if (LoadInst* loadInst = dyn_cast<LoadInst>(Val)) {
      if (GetElementPtrInst* GEPInst = dyn_cast<GetElementPtrInst>(loadInst->getPointerOperand())) {
	Value* OffsetVal = GEPInst->getOperand(GEPInst->getNumOperands() - 1);
	if (ConstantInt* ConstVal = dyn_cast<ConstantInt>(OffsetVal)) {
	  int offset = ConstVal->getValue().getSExtValue();
	  // Confirm vtable load pattern, i.e. GEP vtable, offset
	  // TODO: consider multiple virtual function
	  if (LoadInst* loadClsRef = dyn_cast<LoadInst>(GEPInst->getPointerOperand())) {
	    // Load class reference;
	    Value* ClsRefVal = loadClsRef->getPointerOperand();
	    
	    VTFCS.loadVTF = loadInst;
	    VTFCS.getVTFOffset = GEPInst;
	    VTFCS.loadVT = loadClsRef;
	    VTFCS.clsTy = checkClassRef(ClsRefVal, cha);
	    if (VTFCS.clsTy != nullptr) {
	      // Get the real offset
	      VTFCS.VTFOffset = cha.getOriginalOffset(VTFCS.clsTy, offset);
	      
	      return true;
	    } else  {
	    }
	  }
	}
      } else if (LoadInst* loadVT = dyn_cast<LoadInst>(loadInst->getPointerOperand())) {
	if (AddrSpaceCastInst* asCast = dyn_cast<AddrSpaceCastInst>(loadVT->getPointerOperand())) {
	  VTFCS.loadVTF = loadInst;
	  VTFCS.getVTFOffset = nullptr;
	  VTFCS.loadVT = loadVT;
	  VTFCS.clsTy = checkClassRef(asCast->getPointerOperand(), cha);
	  if (VTFCS.clsTy != nullptr) {
	    // Get the real offset  
	    VTFCS.VTFOffset = 2;
	    
	    return true;
	  } else {
	    Function* Func = asCast->getParent()->getParent();  
	    Value* PtrOp = asCast->getPointerOperand();
            if (Argument* Arg = dyn_cast<Argument>(PtrOp)) {
	      int idx = 0;
              for (; idx < Func->arg_size(); idx ++) {
                if (Arg == Func->getArg(idx))
                  break;
              }

	      if (Type* ClsTy = cha.getArgClassTypeByName(Func, idx)) {
                VTFCS.VTFOffset = 2;
		VTFCS.clsTy = dyn_cast<StructType>(ClsTy);

		return true;
	      } else if (Type* ClsTy = cha.getArgClassTypeByCast(Func, idx)) {
		VTFCS.VTFOffset = 2;
                VTFCS.clsTy = dyn_cast<StructType>(ClsTy);

		return true;
	      }
	    }
	  }
	}
      }
    }
    
    return false;
  }
  
  // Analyze the function pointer 
  void VirtCallAnalysis::analyzeCallSite(CallInst* callInst, Module& M, CHAInfo& cha) {
    Value* CallVal = callInst->getCalledOperand();
    if (dyn_cast<Function>(CallVal))
      // Only function pointer can be considered
      return;

    // Handle call function pointer
    VTFCallSite VTFCS;
    if (checkVTableInsts(CallVal, cha, VTFCS)) {
      VTFCSs[callInst] = VTFCS;
    }
  }
  
  void VirtCallAnalysis::analyze(Module& M, CHAInfo& cha) {
    for (Function& func : M) {
      for (BasicBlock& BB : func) {
	for (Instruction& I : BB) {
	  if (CallInst* callInst = dyn_cast<CallInst>(&I))
	    analyzeCallSite(callInst, M, cha);
	}
      }
    }
  }

  void VirtCallAnalysis::retrieveVFInfos(Module& M, CHAInfo& cha) {
    for (auto& pair : VTFCSs) {
      CallInst* callInst = pair.first;
      VTFCallSite& VTFCS = VTFCSs[callInst];
      if (VTFCS.WF != nullptr)
	continue;
      
      // Get the wrapper function
      Function* WrapperFunc = cha.getWrapperFunction(VTFCS.clsTy, VTFCS.VTFOffset);
      VTFCS.WF = WrapperFunc;
    }
  }

  void VirtCallAnalysis::replaceVFCalls(llvm::Module& M, CHAInfo& cha) {
    for (auto& pair : VTFCSs) {
      CallInst* callInst = pair.first;
      VTFCallSite& VTFCS = VTFCSs[callInst];
      if (VTFCS.WF == nullptr)
	continue;
      
      // Get the wrapper function
      Function* WrapperFunc = cha.getWrapperFunction(VTFCS.clsTy, VTFCS.VTFOffset);

      // Get the basic block
      BasicBlock* CurrBB = callInst->getParent();

      // Create the new call instruction
      FunctionType* FuncTy = WrapperFunc->getFunctionType();
      auto CalledFunc = callInst->getCalledFunction();
      
      callInst->setCalledFunction(WrapperFunc);

      // erase VF call check 0
      // callInst->eraseFromParent();
      // erase wrapper function
      // WrapperFunc->eraseFromParent();
      
      // Erase the irrelevant instructions
      VTFCS.loadVTF->eraseFromParent();
      if (VTFCS.getVTFOffset != nullptr)
	VTFCS.getVTFOffset->eraseFromParent();
      VTFCS.loadVT->eraseFromParent();
    }
  }

  // Replace the VTable setup in constructors
  void VirtCallAnalysis::replaceVTSetups(Module& M, CHAInfo& cha) {
    // Reset the VTable related store instructions' operands
    for (auto& pair : cha.con2Cls) {
      Function& ConsFunc = * pair.first;
      Type* ClsTy = pair.second;
      
      for (BasicBlock& BB : ConsFunc) {
	for (Instruction& I : BB) {
	  if (StoreInst* storeInst = dyn_cast<StoreInst>(&I)) {
	    if (GEPOperator* GEPOp = dyn_cast<GEPOperator>(storeInst->getValueOperand())) {
	      Value* PtrOp = GEPOp->getPointerOperand();
	      if (cha.IsVTable(PtrOp)) {
		// Reset operand
		int ClsID = cha.GetClassID(ClsTy);
		storeInst->setOperand(0, ConstantInt::get(Type::getInt32Ty(M.getContext()), ClsID));
	      }
	    }
	  }
	}
      }

      ConsFunc.dump();
    }

    // xxx Try to replace everything
    for (Function& Func : M) {
      for (BasicBlock& BB : Func) {
        for (Instruction& I : BB) {
          if (StoreInst* storeInst = dyn_cast<StoreInst>(&I)) {
            if (GEPOperator* GEPOp = dyn_cast<GEPOperator>(storeInst->getValueOperand())) {
              Value* PtrOp = GEPOp->getPointerOperand();
              if (cha.IsVTable(PtrOp)) {
                // Reset operand
                storeInst->setOperand(0, ConstantInt::get(Type::getInt32Ty(M.getContext()), 1));
              }
            }
          }
        }
      }
    }

    // Erase VTable related global values
    for (auto& pair : cha.vt2Cls) {
      Value* VT = pair.first;
      if (GlobalVariable* GV = dyn_cast<GlobalVariable>(VT))
	GV->eraseFromParent();
    }

    // M.dump();
  }
  
  void CUDADeVirt::CHA(Module& M) {
    // Chech through global values to retrieve the class related objects
    for (llvm::GlobalVariable& global : M.globals()) {
      
      if (global.getType()->isStructTy()) {
	global.getType()->print(llvm::outs());
	if (IsStructType(global.getType()))
	  llvm::outs() << " is struct type ";
	llvm::outs() << " \n";
	if (global.hasInitializer()) {
	  llvm::outs() << "Initializer:    ";
	  global.getInitializer()->print(llvm::outs()); 
	  llvm::outs() << "\n";
	}
      }
    }

    ValueSymbolTable& SymTable = M.getValueSymbolTable();
    for (ValueSymbolTable::iterator it = SymTable.begin(); it != SymTable.end(); ++ it) {
      Value* Val = it->second;   
    }

    for (auto& Ty : M.getIdentifiedStructTypes()) {
      if (Ty->isStructTy()) {
	// Ty->print(llvm::outs());
	// llvm::outs() << "  \n ";
      }
    }
  }

  void CUDADeVirt::replaceTypeInValue(Value* Val, ClassTypeMap& clsTypeMap) {
    Type* Ty = Val->getType();
    if (Ty->isPointerTy() && Ty->getNumContainedTypes()) {
      Ty = Ty->getPointerElementType();
    }

    if (!Ty->isStructTy())
      return;

    StructType* OldTy = dyn_cast<StructType>(Ty);
    if (clsTypeMap.find(OldTy) != clsTypeMap.end()) {
      StructType* NewTy = clsTypeMap[OldTy];
      Val->mutateType(NewTy);
    }
  }
  
  void CUDADeVirt::rebuildClassTypes(Module& M, CHAInfo& cha) {
    ClassTypeMap& clsTypeMap = cha.getClassTypeMap();

    for (Function& func : M) {
      const char* funcName_ = llvm::itaniumDemangle(func.getName().str().c_str(), nullptr, nullptr, nullptr);
      
      for (BasicBlock& BB : func) {
	for (Instruction& I : BB) {
	  replaceTypeInValue(&I, clsTypeMap);

	  // Traverse the operands of the value
	  for (unsigned int i = 0; i < I.getNumOperands(); i ++) {
	    Value* op = I.getOperand(i);
	    if (Value* UseVal = dyn_cast<Value>(op)) {
	      replaceTypeInValue(UseVal, clsTypeMap);
	    }
	  }
	}
      }
    }
  }
  
  void CUDADeVirt::apply(Module& M) {
    // Apply chass hierarchy analyss
    CHAInfo cha(M);
    
    // Rebuild class types
    // rebuildClassTypes(M, cha);

    // Analyze the virtual call site
    VirtCallAnalysis vca(M, cha);

    // Retrieve wrapper functions
    vca.retrieveVFInfos(M, cha);

    // Replace virtual calls
    vca.replaceVFCalls(M, cha);

    // Replace the VTable setup in constructors
    vca.replaceVTSetups(M, cha);
  }
};



#if LLVM_VERSION_MAJOR > 11
class HipCUDADVPass
    : public PassInfoMixin<HipCUDADVPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif

#endif
