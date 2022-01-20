

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "spirv.hh"

const std::string OpenCL_STD{"OpenCL.std"};

class SPIRVtype {
  size_t z;

public:
  SPIRVtype(size_t s) : z(s) {}
  virtual ~SPIRVtype(){};
  size_t size() { return z; }
  virtual OCLType ocltype() = 0;
  virtual OCLSpace getAS() { return OCLSpace::Private; }
};

typedef std::map<int32_t, SPIRVtype*> SPIRTypeMap;

class SPIRVtypePOD : public SPIRVtype {
public:
  SPIRVtypePOD(int32_t id, size_t size) : SPIRVtype(size) {}
  virtual ~SPIRVtypePOD(){};
  virtual OCLType ocltype() override { return OCLType::POD; }
};

class SPIRVtypePointer : public SPIRVtype {
  OCLSpace ASpace;

public:
  SPIRVtypePointer(int32_t id, int32_t stor_class, size_t pointerSize)
      : SPIRVtype(pointerSize) {
    switch (stor_class) {
    case (int32_t)spv::StorageClass::CrossWorkgroup:
      ASpace = OCLSpace::Global;
      break;

    case (int32_t)spv::StorageClass::Workgroup:
      ASpace = OCLSpace::Local;
      break;

    case (int32_t)spv::StorageClass::UniformConstant:
      ASpace = OCLSpace::Constant;
      break;

    case (int32_t)spv::StorageClass::Function:
      assert(0 && "should have been handled elsewhere!");
      break;

    default:
      ASpace = OCLSpace::Unknown;
    }
  }
  virtual ~SPIRVtypePointer(){};
  virtual OCLType ocltype() override { return OCLType::Pointer; }
  OCLSpace getAS() override { return ASpace; }
};

class SPIRVinst {
  spv::Op opcode;
  size_t wordCount;
  int32_t word1;
  int32_t word2;
  int32_t word3;
  std::string extra;
  int32_t* orig_stream;

public:
  SPIRVinst(int32_t* stream) {
    orig_stream = stream;
    int32_t word0 = stream[0];
    wordCount = (unsigned)word0 >> 16;
    opcode = (spv::Op)(word0 & 0xFFFF);

    if (wordCount > 1)
      word1 = stream[1];

    if (wordCount > 2)
      word2 = stream[2];

    if (wordCount > 3)
      word3 = stream[3];

    if (opcode == spv::Op::OpEntryPoint) {
      const char* pp = (const char*)(stream + 3);
      extra = pp;
    }

    if (opcode == spv::Op::OpExtInstImport) {
      const char* pp = (const char*)(stream + 2);
      extra = pp;
    }
  }

  bool isKernelCapab() const {
    return (opcode == spv::Op::OpCapability) &&
           (word1 == (int32_t)spv::Capability::Kernel);
  }
  bool isExtIntOpenCL() const { return extra == OpenCL_STD; }
  bool isMemModelOpenCL() const {
    return (opcode == spv::Op::OpMemoryModel) &&
           (word2 == (int32_t)spv::MemoryModel::OpenCL);
  }
  size_t getPointerSize() const {
    if (opcode != spv::Op::OpMemoryModel)
      return 0;
    return (word1 == (int32_t)spv::AddressingModel::Physical64) ? 8 : 4;
  }
  bool isLangOpenCL() const {
    return (opcode == spv::Op::OpSource) &&
           ((word1 == (int32_t)spv::SourceLanguage::OpenCL_C) ||
            (word1 == (int32_t)spv::SourceLanguage::OpenCL_CPP));
  }

  bool isEntryPoint() {
    return (opcode == spv::Op::OpEntryPoint) &&
           (word1 == (int32_t)spv::ExecutionModel::Kernel);
  }
  int32_t entryPointID() { return word2; }
  std::string&& entryPointName() { return std::move(extra); }

  size_t size() const { return wordCount; }
  spv::Op getOpcode() const { return opcode; }

  int32_t getFunctionID() const { return word2; }
  int32_t getFunctionTypeID() const { return orig_stream[4]; }
  int32_t getFunctionRetType() const { return word1; }

  bool isType() const {
    return ((int32_t)opcode >= (int32_t)spv::Op::OpTypeVoid) &&
           ((int32_t)opcode <= (int32_t)spv::Op::OpTypeForwardPointer);
  }
  int32_t getTypeID() const {
    assert(isType());
    return word1;
  }
  bool isFunctionType() const { return (opcode == spv::Op::OpTypeFunction); }
  bool isFunction() const { return (opcode == spv::Op::OpFunction); }

  SPIRVtype* decodeType(SPIRTypeMap& typeMap, size_t pointerSize) {
    if (opcode == spv::Op::OpTypeVoid) {
      return new SPIRVtypePOD(word1, 0);
    }

    if (opcode == spv::Op::OpTypeBool) {
      return new SPIRVtypePOD(word1, 1);
    }

    if (opcode == spv::Op::OpTypeInt) {
      return new SPIRVtypePOD(word1, ((size_t)word2 / 8));
    }

    if (opcode == spv::Op::OpTypeFloat) {
      return new SPIRVtypePOD(word1, ((size_t)word2 / 8));
    }

    if (opcode == spv::Op::OpTypeVector) {
      size_t type_size = typeMap[word2]->size();
      return new SPIRVtypePOD(word1, type_size * orig_stream[3]);
    }

    if (opcode == spv::Op::OpTypeArray) {
      size_t type_size = typeMap[word2]->size();
      return new SPIRVtypePOD(word1, type_size * word3);
    }

    if (opcode == spv::Op::OpTypeStruct) {
      size_t total_size = 0;
      for (size_t i = 2; i < wordCount; ++i) {
        int32_t member_id = orig_stream[i];
        total_size += typeMap[member_id]->size();
      }
      return new SPIRVtypePOD(word1, total_size);
    }

    if (opcode == spv::Op::OpTypePointer) {
      // structs or vectors passed by value are represented in LLVM IR / SPIRV
      // by a pointer with "byval" keyword; handle them here
      if (word2 == (int32_t)spv::StorageClass::Function) {
        int32_t pointee = word3;
        size_t pointee_size = typeMap[pointee]->size();
        return new SPIRVtypePOD(word1, pointee_size);

      } else
        return new SPIRVtypePointer(word1, word2, pointerSize);
    }

    return nullptr;
  }

  OCLFuncInfo* decodeFunctionType(SPIRTypeMap& typeMap, size_t pointerSize) {
    assert(opcode == spv::Op::OpTypeFunction);

    OCLFuncInfo* fi = new OCLFuncInfo;

    int32_t ret_id = word2;
    auto it = typeMap.find(ret_id);
    assert(it != typeMap.end());
    fi->RetTypeInfo.Type = it->second->ocltype();
    fi->RetTypeInfo.Size = it->second->size();
    fi->RetTypeInfo.Space = it->second->getAS();

    size_t n_args = wordCount - 3;
    if (n_args > 0) {
      fi->ArgTypeInfo.resize(n_args);
      for (size_t i = 0; i < n_args; ++i) {
        int32_t type_id = orig_stream[i + 3];
        auto it = typeMap.find(type_id);
        assert(it != typeMap.end());
        fi->ArgTypeInfo[i].Type = it->second->ocltype();
        fi->ArgTypeInfo[i].Size = it->second->size();
        fi->ArgTypeInfo[i].Space = it->second->getAS();
      }
    }

    return fi;
  }
};

class SPIRVmodule {
  std::map<int32_t, std::string> entryPoints;
  SPIRTypeMap typeMap;
  OCLFuncInfoMap functionTypeMap;
  std::map<int32_t, int32_t> entryToFunctionTypeIDMap;

  bool languageCL;
  bool memModelCL;
  bool kernelCapab;
  bool extIntOpenCL;
  bool headerOK;
  bool parseOK;

public:
  ~SPIRVmodule() {
    for (auto I : typeMap) {
      delete I.second;
    }
  }

  bool valid() {
    return headerOK && kernelCapab && extIntOpenCL && languageCL &&
           memModelCL && parseOK;
  }

  bool parseSPIRV(int32_t* stream, size_t numWords) {
    int32_t* p = stream;

    kernelCapab = false;
    extIntOpenCL = false;
    headerOK = false;
    languageCL = false;
    memModelCL = false;
    parseOK = false;

    if (*p != spv::MagicNumber)
      return false;
    ++p;

    if ((*p != spv::Version10) && (*p != spv::Version11))
      return false;
    ++p;

    // GENERATOR
    ++p;

    // BOUND
    int32_t bound = *p;
    ++p;

    // RESERVED
    if (*p != 0)
      return false;
    ++p;

    headerOK = true;

    // INSTRUCTION STREAM
    parseOK = parseInstructionStream(p, (numWords - 5));
    return valid();
  }

  bool fillModuleInfo(OpenCLFunctionInfoMap& moduleMap) {
    if (!valid())
      return false;

    for (auto i : entryPoints) {
      int32_t EntryPointID = i.first;
      auto ft = entryToFunctionTypeIDMap.find(EntryPointID);
      assert(ft != entryToFunctionTypeIDMap.end());
      auto fi = functionTypeMap.find(ft->second);
      assert(fi != functionTypeMap.end());
      moduleMap.emplace(std::make_pair(i.second, fi->second));
    }

    return true;
  }

private:
  bool parseInstructionStream(int32_t* stream, size_t numWords) {
    int32_t* p = stream;
    size_t pointerSize = 0;
    while (numWords > 0) {
      SPIRVinst inst(p);

      if (inst.isKernelCapab())
        kernelCapab = true;

      if (inst.isExtIntOpenCL())
        extIntOpenCL = true;

      if (inst.isMemModelOpenCL()) {
        memModelCL = true;
        pointerSize = inst.getPointerSize();
        assert(pointerSize > 0);
      }

      if (inst.isLangOpenCL())
        languageCL = true;

      if (inst.isEntryPoint()) {
        entryPoints.emplace(
            std::make_pair(inst.entryPointID(), inst.entryPointName()));
      }

      if (inst.isType()) {
        if (inst.isFunctionType())
          functionTypeMap.emplace(std::make_pair(
              inst.getTypeID(), inst.decodeFunctionType(typeMap, pointerSize)));
        else
          typeMap.emplace(std::make_pair(
              inst.getTypeID(), inst.decodeType(typeMap, pointerSize)));
      }

      if (inst.isFunction() &&
          (entryPoints.find(inst.getFunctionID()) != entryPoints.end())) {
        // ret type must be void
        auto retty = typeMap.find(inst.getFunctionRetType());
        assert(retty != typeMap.end());
        assert(typeMap[inst.getFunctionRetType()]->size() == 0);

        entryToFunctionTypeIDMap.emplace(
            std::make_pair(inst.getFunctionID(), inst.getFunctionTypeID()));
      }

      numWords -= inst.size();
      p += inst.size();
    }

    return true;
  }
};

bool parseSPIR(int32_t* stream, size_t numWords,
               OpenCLFunctionInfoMap& output) {
  SPIRVmodule Mod;
  if (!Mod.parseSPIRV(stream, numWords))
    return false;
  return Mod.fillModuleInfo(output);
}
