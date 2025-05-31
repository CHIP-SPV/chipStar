/*
 * Copyright (c) 2021-23 chipStar developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <regex>

#include "common.hh"
#include "spirv.hh"
#include "logging.hh"
#include "Utils.hh"

const std::string OpenCLStd{"OpenCL.std"};

// TODO: Refactor. Separate type for ID values for avoiding mixing up
//       them with instruction words.
using InstWord = uint32_t;

static std::string_view parseLiteralString(const InstWord *WordBegin,
                                           size_t NumWords);

/// Represents alignment value which is non-zero 2**N integer.
class Alignment {
  size_t Val_ = 1;

public:
  Alignment() = default;
  Alignment(size_t Val) : Val_(Val) {
    assert(Val > 0 && "Must be non-zero");
    assert(((Val - 1) & Val) == 0 && "Not power of two.");
  }
  operator size_t() const { return Val_; }
};

class SPIRVtype {
  InstWord Id_;
  size_t Size_;
  /// Required alignment. For primitive types it is smallest
  /// power-of-two => Size_. For aggregates, it is the largest
  /// required alignment of its members.
  Alignment Align_;

public:
  /// Type with requested alignment.
  SPIRVtype(InstWord Id, size_t Size, Alignment AlignVal)
      : Id_(Id), Size_(Size), Align_(AlignVal) {}
  /// Type with a power-of-tow alignment deducted from 'Size'.
  SPIRVtype(InstWord Id, size_t Size)
      : SPIRVtype(Id, Size, roundUpToPowerOfTwo(Size)) {}
  virtual ~SPIRVtype(){};
  size_t size() { return Size_; }
  size_t alignment() const { return Align_; }
  InstWord id() { return Id_; }
  virtual SPVTypeKind typeKind() = 0;
  virtual SPVStorageClass getSC() { return SPVStorageClass::Private; }
};

typedef std::map<InstWord, SPIRVtype *> SPIRTypeMap;

class SPIRVtypePOD : public SPIRVtype {
public:
  SPIRVtypePOD(InstWord Id, size_t Size, Alignment AlignVal)
      : SPIRVtype(Id, Size, AlignVal) {}
  SPIRVtypePOD(InstWord Id, size_t Size) : SPIRVtype(Id, Size) {}
  virtual ~SPIRVtypePOD(){};
  virtual SPVTypeKind typeKind() override { return SPVTypeKind::POD; }
};

class SPIRVtypeArray : public SPIRVtypePOD {
  size_t EltCount_;

public:
  SPIRVtypeArray(InstWord Id, SPIRVtype *EltType, size_t ElementCount)
      : SPIRVtypePOD(
            Id, ElementCount * roundUp(EltType->size(), EltType->alignment()),
            EltType->alignment()),
        EltCount_(ElementCount) {}
  size_t elementCount() const { return EltCount_; }
};

class SPIRVtypeOpaque : public SPIRVtype {
  std::string Name_;

public:
  SPIRVtypeOpaque(InstWord Id, std::string_view Name)
      : SPIRVtype(Id, 0), Name_(Name) {} // Opaque types are unsized.
  virtual ~SPIRVtypeOpaque(){};
  virtual SPVTypeKind typeKind() override { return SPVTypeKind::Opaque; }
};

class SPIRVtypeImage : public SPIRVtype {
public:
  SPIRVtypeImage(InstWord Id) : SPIRVtype(Id, 0) {}
  virtual ~SPIRVtypeImage(){};
  virtual SPVTypeKind typeKind() override { return SPVTypeKind::Image; }
  virtual SPVStorageClass getSC() override { return SPVStorageClass::Unknown; }
};

class SPIRVtypeSampler : public SPIRVtype {
public:
  SPIRVtypeSampler(InstWord Id) : SPIRVtype(Id, 0) {}
  virtual ~SPIRVtypeSampler(){};
  virtual SPVTypeKind typeKind() override { return SPVTypeKind::Sampler; }
  virtual SPVStorageClass getSC() override {
    return SPVStorageClass::UniformConstant;
  }
};

class SPIRVtypePointer : public SPIRVtype {
  SPVStorageClass StorageClass_;
  InstWord PointeeTypeID_;

public:
  SPIRVtypePointer(InstWord Id, InstWord StorClass, size_t PointerSize,
                   InstWord PointeeTypeID)
      : SPIRVtype(Id, PointerSize), PointeeTypeID_(PointeeTypeID) {
    switch (StorClass) {
    case (int32_t)spv::StorageClassCrossWorkgroup:
      StorageClass_ = SPVStorageClass::CrossWorkgroup;
      break;

    case (int32_t)spv::StorageClassWorkgroup:
      StorageClass_ = SPVStorageClass::Workgroup;
      break;

    case (int32_t)spv::StorageClassUniformConstant:
      StorageClass_ = SPVStorageClass::UniformConstant;
      break;

    case (InstWord)spv::StorageClassFunction:
      StorageClass_ = SPVStorageClass::Private;
      break;

    default:
      StorageClass_ = SPVStorageClass::Unknown;
    }
  }
  virtual ~SPIRVtypePointer(){};
  virtual SPVTypeKind typeKind() override { return SPVTypeKind::Pointer; }
  SPVStorageClass getSC() override { return StorageClass_; }
  InstWord getPointeeTypeID() const { return PointeeTypeID_; }
};

class SPIRVConstant {
  std::vector<InstWord> ConstantWords_;

public:
  SPIRVConstant(SPIRVtype *Type, size_t NumConstWords,
                const InstWord *ConstWords) {
    ConstantWords_.insert(ConstantWords_.end(), ConstWords,
                          ConstWords + NumConstWords);
  }

  template <typename T> T interpretAs() const {
    assert(false && "Undefined accessor!");
  };
};

// Explicit specialization of SPIRVConstant::interpretAs for uint64_t
template <>
inline uint64_t SPIRVConstant::interpretAs<uint64_t>() const {
  assert(ConstantWords_.size() > 0 && "Invalid constant word count.");
  assert(ConstantWords_.size() <= 2 && "Constant may not fit to uint64_t.");
  if (ConstantWords_.size() == 1)
    return static_cast<uint32_t>(ConstantWords_[0]);
  // Copy the value in order to satisfy alignment requirement of the type.
  return copyAs<uint64_t>(ConstantWords_.data());
}

typedef std::map<InstWord, SPIRVConstant *> SPIRVConstMap;

// Parses and checks SPIR-V header. Sets word buffer pointer to poin
// past the header and updates NumWords count to exclude header words.
// Return false if there is an error in the header. Otherwise, return
// true.
static bool parseHeader(const InstWord *&WordBuffer, size_t &NumWords) {
  if (*WordBuffer != spv::MagicNumber) {
    logError("Incorrect SPIR-V magic number.");
    return false;
  }
  ++WordBuffer;

  // Jump over VERSION, GENERATOR and BOUND words.
  WordBuffer += 3;

  // RESERVED
  if (*WordBuffer != 0) {
    logError("Invalid SPIR-V: Reserved word is not 0.");
    return false;
  }
  ++WordBuffer;

  NumWords -= 5;
  return true;
}

class SPIRVinst {
  spv::Op Opcode_;
  size_t WordCount_;
  const InstWord *Words_;
  std::string_view Extra_;

public:
  SPIRVinst(const InstWord *Stream) {
    Words_ = Stream;
    WordCount_ = Words_[0] >> 16;
    Opcode_ = (spv::Op)(Words_[0] & 0xFFFF);

    if (Opcode_ == spv::Op::OpEntryPoint) {
      const char *Pp = (const char *)(Stream + 3);
      Extra_ = Pp;
    }

    if (Opcode_ == spv::Op::OpExtInstImport ||
        Opcode_ == spv::Op::OpTypeOpaque || Opcode_ == spv::Op::OpName)
      Extra_ = (const char *)(Stream + 2);
  }

  /// Get word in the instructions. getWord(0) gives word for the
  /// instruction length and opcode.
  const InstWord &getWord(unsigned Idx) const {
    assert(Idx < WordCount_ && "Out-of-bounds index!");
    return Words_[Idx];
  }

  /// Get operand of the instruction
  ///
  /// Indexing starts past instruction's opcode and result ID and
  /// result type words.
  const InstWord &getOperand(unsigned Idx) const {
    bool HasResultID;
    bool HasResultType;
    HasResultAndType(getOpcode(), &HasResultID, &HasResultType);
    return getWord(Idx + 1 + HasResultID + HasResultType);
  }

  bool isKernelCapab() const {
    return (Opcode_ == spv::Op::OpCapability) &&
           (getWord(1) == (InstWord)spv::CapabilityKernel);
  }
  bool isExtIntOpenCL() const { return Extra_ == OpenCLStd; }
  bool isMemModelOpenCL() const {
    return (Opcode_ == spv::Op::OpMemoryModel) &&
           (getWord(2) == (InstWord)spv::MemoryModelOpenCL);
  }
  size_t getPointerSize() const {
    if (Opcode_ != spv::Op::OpMemoryModel)
      return 0;
    return (getWord(1) == (InstWord)spv::AddressingModelPhysical64) ? 8 : 4;
  }
  bool isLangOpenCL() const {
    return (Opcode_ == spv::Op::OpSource) &&
           ((getWord(1) == (InstWord)spv::SourceLanguageOpenCL_C) ||
            (getWord(1) == (InstWord)spv::SourceLanguageOpenCL_CPP));
  }

  bool isEntryPoint() const {
    return (Opcode_ == spv::Op::OpEntryPoint) &&
           (getWord(1) == (InstWord)spv::ExecutionModelKernel);
  }
  InstWord entryPointID() { return getWord(2); }
  std::string_view entryPointName() const { return Extra_; }

  size_t size() const { return WordCount_; }
  spv::Op getOpcode() const { return Opcode_; }
  template <spv::Op Opcode> bool isa() const { return Opcode == getOpcode(); }

  InstWord getFunctionID() const { return getWord(2); }
  InstWord getFunctionTypeID() const { return Words_[4]; }
  InstWord getFunctionRetType() const { return getWord(1); }

  bool isType() const {
    return ((InstWord)Opcode_ >= (InstWord)spv::Op::OpTypeVoid) &&
           ((InstWord)Opcode_ <= (InstWord)spv::Op::OpTypeForwardPointer);
  }
  InstWord getTypeID() const {
    assert(isType());
    return getWord(1);
  }

  bool hasResultType() const {
    bool Ignored, HasType;
    spv::HasResultAndType(Opcode_, &Ignored, &HasType);
    return HasType;
  }

  InstWord getResultTypeID() const {
    assert(hasResultType() && "Instruction does not have a result type!");
    return getWord(1);
  }

  bool hasResultID() const {
    bool HasResult, Ignored;
    spv::HasResultAndType(Opcode_, &HasResult, &Ignored);
    return HasResult;
  }

  InstWord getResultID() const {
    assert(hasResultID() && "Instruction does not have a result operand!");
    return hasResultType() ? getWord(2) : getWord(1);
  }

  /// Return true if the instruction is a kind that may forward
  /// reference instructions (OpName, OpDecorate, ...)
  ///
  /// The reference (Result ID) is returned via 'TargetResultID' if it
  /// is not nullptr;
  bool isForwardReferencing(InstWord *TargetResultID = nullptr) const {
    switch (getOpcode()) {
      // NOTE: this switch is not exhaustive. Extended on-demand basis.
    default:
      return false;
    case spv::OpName:
    case spv::OpDecorate:
      if (TargetResultID)
        *TargetResultID = getWord(1);
      return true;
      // TODO: OpEntryFunction
    }
  }

  bool isFunctionType() const { return (Opcode_ == spv::Op::OpTypeFunction); }
  bool isFunction() const { return (Opcode_ == spv::Op::OpFunction); }
  bool isConstant() const { return (Opcode_ == spv::Op::OpConstant); }

  bool isGlobalVariable() const {
    if (getOpcode() == spv::OpVariable &&
        getWord(3) != spv::StorageClassFunction)
      return true;
    return false;
  }

  // Return true if the instruction is an OpName.
  bool isName() const { return Opcode_ == spv::Op::OpName; }

  std::string_view getName() const {
    assert(isName() && "Not an OpName!");
    return Extra_;
  }

  // Return true if the instruction is the given Decoration.
  bool isDecoration(spv::Decoration Dec) const {
    return Opcode_ == spv::Op::OpDecorate && getWord(2) == (InstWord)Dec;
  }

  bool isExtension() const { return Opcode_ == spv::Op::OpExtension; }

  std::string_view getExtension() const {
    assert(isExtension() && "Not an OpExtension!");
    auto StrSize = size() - /* offset to the string: */ 1;
    return parseLiteralString(&getWord(1), StrSize);
  }

  SPIRVtype *decodeType(SPIRTypeMap &TypeMap, SPIRVConstMap &ConstMap,
                        size_t PointerSize) {
    if (Opcode_ == spv::Op::OpTypeVoid) {
      return new SPIRVtypePOD(getWord(1), 0);
    }

    if (Opcode_ == spv::Op::OpTypeBool) {
      return new SPIRVtypePOD(getWord(1), 1);
    }

    if (Opcode_ == spv::Op::OpTypeInt) {
      return new SPIRVtypePOD(getWord(1), ((size_t)getWord(2) / 8));
    }

    if (Opcode_ == spv::Op::OpTypeFloat) {
      return new SPIRVtypePOD(getWord(1), ((size_t)getWord(2) / 8));
    }

    if (Opcode_ == spv::Op::OpTypeVector) {
      auto Type = TypeMap[getWord(2)];
      if (!Type) {
        logWarn("SPIR-V Parser: getWord(2) {} not found in type map",
                getWord(2));
        return nullptr;
      }
      size_t TypeSize = Type->size();
      return new SPIRVtypePOD(getWord(1), TypeSize * getWord(3));
    }

    if (Opcode_ == spv::Op::OpTypeArray) {
      auto EltType = TypeMap[getWord(2)];
      if (!EltType) {
        logWarn("SPIR-V Parser: getWord(2) {} not found in type map",
                getWord(2));
        return nullptr;
      }
      // Compute actual element size due padding for meeting the
      // alignment requirements.  C analogy as example: 'struct {int
      // a; char b; }' takes 8 bytes per element in the array.
      //
      auto *EltCountOperand = ConstMap[getWord(3)];
      if (!EltCountOperand) {
        logWarn("SPIR-V Parser: Could not parse OpConstant "
                "operand.");
        return nullptr;
      }
      auto EltCount = EltCountOperand->interpretAs<uint64_t>();
      return new SPIRVtypeArray(getWord(1), EltType, EltCount);
    }

    if (Opcode_ == spv::Op::OpTypeStruct) {
      size_t TotalSize = 0;
      Alignment MaxAlignment;
      for (size_t i = 2; i < WordCount_; ++i) {
        InstWord MemberId = getWord(i);

        auto Type = TypeMap[MemberId];
        if (!Type) {
          logWarn("SPIR-V Parser: MemberId {} not found in type map", MemberId);
          continue;
        }
        // Compute actual size as in spv::Op::OpTypeArray branch
        // except don't account the tail padding. C analogy as
        // example: 'struct { char a; int b; char c}' takes 9 bytes.
        size_t MemberAlignment = Type->alignment();
        TotalSize = roundUp(TotalSize, MemberAlignment);
        TotalSize += Type->size();
        if (MemberAlignment > MaxAlignment)
          MaxAlignment = MemberAlignment;
      }
      return new SPIRVtypePOD(getWord(1), TotalSize, MaxAlignment);
    }

    if (Opcode_ == spv::Op::OpTypeOpaque) {
      return new SPIRVtypeOpaque(getWord(1), Extra_);
    }

    if (Opcode_ == spv::Op::OpTypeImage) {
      return new SPIRVtypeImage(getWord(1));
    }

    if (Opcode_ == spv::Op::OpTypeSampler) {
      return new SPIRVtypeSampler(getWord(1));
    }

    if (Opcode_ == spv::Op::OpTypePointer)
      return new SPIRVtypePointer(getWord(1), getWord(2), PointerSize,
                                  getWord(3));

    if (Opcode_ == spv::Op::OpTypeFunction)
      // Not a correct object for function type but close. We currently
      // don't need dedicated class for it.
      return new SPIRVtypeOpaque(getWord(1), "<some-function>");

    return nullptr;
  }

  SPIRVConstant *decodeConstant(SPIRTypeMap &TypeMap) const {
    assert(isConstant());
    assert(WordCount_ >= 4 && "Invalid OpConstant word count!");

    if (auto *Type = TypeMap[getWord(1)])
      return new SPIRVConstant(Type, WordCount_ - 3, &getWord(3));

    logWarn("SPIR-V Parser: Missing type declaration for a constant");
    return nullptr;
  }
};

static std::string_view parseLiteralString(const InstWord *WordBegin,
                                           size_t NumWords) {
  const auto *ByteBegin = (const char *)WordBegin;
  const auto *LastWord = ByteBegin + (NumWords - 1) * sizeof(InstWord);
  auto LastByte = LastWord[3];
  // String literals are nul-terminated [SPIR-V 2.2.1 Instructions].
  assert(LastByte == '\0' && "Missing nul-termination.");
  return std::string_view(ByteBegin);
}

static std::string_view parseLinkageAttributeName(const SPIRVinst &Inst) {
  assert(Inst.isDecoration(spv::DecorationLinkageAttributes));
  auto StrSize = Inst.size() -
                 /* offset to the string: */ 3 -
                 /* linkage type: */ 1;
  return parseLiteralString(&Inst.getWord(3), StrSize);
}

static std::string_view parseLinkageAttributeName(const SPIRVinst *Inst) {
  return parseLinkageAttributeName(*Inst);
}

static spv::LinkageType parseLinkageAttributeType(const SPIRVinst &Inst) {
  assert(Inst.isDecoration(spv::DecorationLinkageAttributes));
  return static_cast<spv::LinkageType>(Inst.getWord(Inst.size() - 1));
}

static spv::FunctionParameterAttribute
parseFunctionParameterAttribute(const SPIRVinst &Inst) {
  assert(Inst.isDecoration(spv::DecorationFuncParamAttr));
  return static_cast<spv::FunctionParameterAttribute>(Inst.getWord(3));
}

static IteratorRange<const InstWord *> getWordRange(const InstWord *Begin,
                                                    size_t NumWords) {
  return IteratorRange<const InstWord *>(Begin, Begin + NumWords);
}

class SPIRVmodule {
  std::map<InstWord, std::string> EntryPoints_;
  SPIRTypeMap TypeMap_;
  SPIRVConstMap ConstMap_;
  SPVFuncInfoMap KernelInfoMap_;
  // Records of result IDs decorated with ByVal function parameter attribute.
  std::map<InstWord, bool> ByValParams_;
  std::unordered_map<InstWord, std::unique_ptr<SPIRVinst>> IdToInstMap_;
  /// Names of globals and functions.
  std::map<InstWord, std::string_view> LinkNames_;
  std::map<std::string_view, std::vector<std::pair<uint16_t, uint16_t>>>
      SpilledArgAnnotations_;

  // This flag indicates if the module is known not to have indirect
  // global buffer accesses (IGBA) in any kernel. This is told by a
  // magic variable created by HipIGBADetectorPass. Defaults to false
  // in case the variable is not found.
  bool HasNoIGBAs_ = false;

  bool MemModelCL_;
  bool KernelCapab_;
  bool ExtIntOpenCL_;
  bool HeaderOK_;
  bool ParseOK_;

public:
  ~SPIRVmodule() {
    for (auto I : TypeMap_)
      delete I.second;
    for (auto I : ConstMap_)
      delete I.second;
  }

  bool valid() {
    bool AllOk = true;
    auto Check = [&](bool Cond, const char *ErrMsg) {
      if (!Cond)
        logError(ErrMsg);
      AllOk &= Cond;
    };

    Check(HeaderOK_, "Invalid SPIR-V header.");
    // TODO: Temporary. With these check disabled the simple_kernel
    //       runs successfully on OpenCL backend at least. Note that we are
    //       passing invalid SPIR-V binary.
    // Check(KernelCapab_, "Kernel capability missing.");
    // Check(ExtIntOpenCL_, "Missing extended OpenCL instructions.");
    Check(MemModelCL_, "Incorrect memory model.");
    Check(ParseOK_, "An error encountered during parsing.");
    return AllOk;
  }

  bool analyzeSPIRV(const InstWord *Stream, size_t NumWords) {
    KernelCapab_ = false;
    ExtIntOpenCL_ = false;
    HeaderOK_ = false;
    MemModelCL_ = false;
    ParseOK_ = false;
    HeaderOK_ = parseHeader(Stream, NumWords);

    // INSTRUCTION STREAM
    ParseOK_ = parseInstructionStream(Stream, NumWords);
    return valid();
  }

  bool fillModuleInfo(SPVModuleInfo &ModuleInfo) {
    if (!valid())
      return false;

    for (auto i : EntryPoints_) {
      InstWord EntryPointID = i.first;
      std::string_view KernelName = i.second;
      auto Fi = KernelInfoMap_.find(EntryPointID);
      assert(Fi != KernelInfoMap_.end());
      auto FnInfo = Fi->second;

      if (SpilledArgAnnotations_.count(KernelName)) {
        FnInfo->HasByRefArgs_ = true;
        for (auto &Kv : SpilledArgAnnotations_[KernelName]) {
          FnInfo->ArgTypeInfo_[Kv.first].Kind = SPVTypeKind::PODByRef;
          FnInfo->ArgTypeInfo_[Kv.first].Size = Kv.second;
        }
      }

      ModuleInfo.FuncInfoMap.emplace(std::make_pair(i.second, FnInfo));
    }
    KernelInfoMap_.clear();

    ModuleInfo.HasNoIGBAs = HasNoIGBAs_;

    return true;
  }

private:
  std::string_view getLinkNameOr(const SPIRVinst *Inst,
                                 std::string_view OrValue) const {
    if (!Inst->hasResultID())
      return OrValue;
    auto It = LinkNames_.find(Inst->getResultID());
    return It != LinkNames_.end() ? It->second : OrValue;
  }

  const SPIRVinst *getInstruction(InstWord ID) const {
    auto It = IdToInstMap_.find(ID);
    return It != IdToInstMap_.end() ? It->second.get() : nullptr;
  }

  void processKernelParameter(const SPIRVinst &Inst, SPVFuncInfo &FuncInfo) {
    // Record kernel parameter size for kernel argument setters in the
    // backends.
    auto ParamTypeID = Inst.getWord(1);
    assert(TypeMap_.count(ParamTypeID) &&
           "Can't calculate parameter size due to missing type info!");
    SPIRVtype *ParamType = TypeMap_[ParamTypeID];
    size_t ParamSize = 0;
    SPVTypeKind TypeKind = SPVTypeKind::Unknown;
    if (ByValParams_.count(Inst.getResultID())) {
      // ByVal attribute may only be attached on pointer parameters.
      auto *PtrType = static_cast<SPIRVtypePointer *>(ParamType);
      SPIRVtype *PointeeType = TypeMap_[PtrType->getPointeeTypeID()];
      assert(PointeeType && "Can't calculate parameter size due to missing "
                            "pointee type info!");
      // Backends treat the ByVal attributes pointer parameters as POD.
      ParamSize = PointeeType->size();
      TypeKind = SPVTypeKind::POD;
    } else {
      ParamSize = ParamType->size();
      TypeKind = ParamType->typeKind();
    }
    FuncInfo.ArgTypeInfo_.emplace_back(
        SPVArgTypeInfo{TypeKind, ParamType->getSC(), ParamSize});
  }

  bool parseInstructionStream(const InstWord *Stream, size_t NumWords) {
    const InstWord *StreamIntPtr = Stream;
    size_t PointerSize = 0;
    InstWord CurrentKernelID = 0;
    SPVFuncInfo *CurrentKernelInfo = nullptr;
    while (NumWords > 0) {
      SPIRVinst TempInst(StreamIntPtr);
      auto *Inst = &TempInst;

      if (Inst->hasResultID()) {
        Inst = new SPIRVinst(StreamIntPtr);
        IdToInstMap_[Inst->getResultID()].reset(Inst);
      }

      if (Inst->isKernelCapab())
        KernelCapab_ = true;

      if (Inst->isExtIntOpenCL())
        ExtIntOpenCL_ = true;

      if (Inst->isMemModelOpenCL()) {
        MemModelCL_ = true;
        PointerSize = Inst->getPointerSize();
        assert(PointerSize > 0);
      }

      if (Inst->isEntryPoint()) {
        EntryPoints_.emplace(
            std::make_pair(Inst->entryPointID(), Inst->entryPointName()));
      }

      if (Inst->isType())
        TypeMap_.emplace(
            std::make_pair(Inst->getTypeID(),
                           Inst->decodeType(TypeMap_, ConstMap_, PointerSize)));

      if (Inst->isFunction() && EntryPoints_.count(Inst->getFunctionID())) {
        CurrentKernelID = Inst->getFunctionID();
        assert(!KernelInfoMap_.count(CurrentKernelID) &&
               "Overwriting existing kernel function info!");
        auto FnInfo = std::make_shared<SPVFuncInfo>();
        KernelInfoMap_[CurrentKernelID] = FnInfo;
        CurrentKernelInfo = FnInfo.get();

        // ret type must be void
        auto Retty = TypeMap_.find(Inst->getFunctionRetType());
        assert(Retty != TypeMap_.end());
        assert(TypeMap_[Inst->getFunctionRetType()]->size() == 0);
      }

      if (Inst->isa<spv::OpFunctionParameter>() && CurrentKernelInfo)
        processKernelParameter(*Inst, *CurrentKernelInfo);

      if (Inst->isa<spv::OpFunctionEnd>())
        CurrentKernelInfo = nullptr;

      if (Inst->isConstant()) {
        auto *Const = Inst->decodeConstant(TypeMap_);
        ConstMap_.emplace(std::make_pair(Inst->getResultID(), Const));
      }

      if (Inst->isDecoration(spv::DecorationLinkageAttributes)) {
        auto TargetID = Inst->getWord(1);
        auto LinkName = parseLinkageAttributeName(Inst);
        LinkNames_[TargetID] = LinkName;
      }

      if (Inst->isDecoration(spv::DecorationFuncParamAttr)) {
        auto Attr = parseFunctionParameterAttribute(*Inst);
        if (Attr == spv::FunctionParameterAttributeByVal) {
          auto TargetID = Inst->getWord(1);
          ByValParams_[TargetID] = true;
        }
      }

      if (Inst->isGlobalVariable()) {
        auto Name = getLinkNameOr(Inst, "");
        auto SpillArgAnnotation = std::string_view(ChipSpilledArgsVarPrefix);
        if (startsWith(Name, SpillArgAnnotation)) {
          auto KernelName = Name.substr(SpillArgAnnotation.size());
          auto &SpillAnnotation = SpilledArgAnnotations_[KernelName];
          // Get initializer operand.
          auto *Init = getInstruction(Inst->getWord(4));
          assert(Init && "Annotation variable is missing an initializer.");
          // Init is known to be OpConstantComposite of char array.
          auto *Type = TypeMap_[Init->getResultTypeID()];
          assert(Type && dynamic_cast<SPIRVtypeArray *>(Type) &&
                 "Could not type for result ID.");
          auto *ArrayType = static_cast<SPIRVtypeArray *>(Type);
          auto ArrLen = ArrayType->elementCount();
          // Iterate constituents.
          for (auto EltID : getWordRange(&Init->getWord(3), ArrLen)) {
            auto *ConstInt = getInstruction(EltID); // OpConstant
            uint32_t Annotation = ConstInt->getWord(3);
            uint16_t ArgIndex = Annotation & 0xffff;
            uint16_t ArgSize = Annotation >> 16u;
            SpillAnnotation.push_back(std::make_pair(ArgIndex, ArgSize));
          }
        }

        // A magic variable created by HipIGBADetector.cpp.
        if (Name == "__chip_module_has_no_IGBAs") {
          // Get initializer operand.
          auto *Init = getInstruction(Inst->getWord(4));
          // Init is known to be 8-bit unsigned constant.
          HasNoIGBAs_ = Init->getWord(3);
        }
      }

      NumWords -= Inst->size();
      StreamIntPtr += Inst->size();
    }

    return true;
  }
};

using IdMapT = std::unordered_map<InstWord, InstWord>;
using IdSetT = std::unordered_set<InstWord>;
enum FilterAction { Error = 0, Keep, Drop, Replace };

static FilterAction
workaroundLlvmSpirvIssue2008(const SPIRVinst &Insn,
                             std::vector<InstWord> &ReplacementInsn,
                             IdMapT &ResultIdMap, IdSetT &SampledImgs) {
  // Workaround issue of
  // https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/2008
  // that occurs with llvm-spirv-16 which produces invalid SPIR-V like
  // the following:
  //
  //   %TempSampledImage = OpSampledImage %14 %i %s
  //                 %16 = OpBitcast %14 %TempSampledImage
  //               %call = OpImageSampleExplicitLod %v4float %16 %c Lod %float_0
  //
  // The %TempSampledImage may only appear as operand to image lookups
  // and queries and bitcasts are not one of them. Additionally,
  // OpTypeSampledImage is not allowed type for bitcast.
  //
  // Fix by dropping the bitcast and passing its input operand to users.

  if (Insn.isa<spv::OpSampledImage>()) {
    // Track result type. We use it to detect invalid bitcast of sampled image.
    SampledImgs.insert(Insn.getResultID());
    return FilterAction::Keep;
  }

  if (Insn.isa<spv::OpBitcast>()) {
    if (SampledImgs.count(Insn.getOperand(0))) {
      // Invalid bitcast of a sampled image. Drop it and replace users of
      // the bitcast with its input operand.
      ResultIdMap[Insn.getResultID()] = Insn.getOperand(0);
      return FilterAction::Drop;
    }
    return FilterAction::Keep;
  }

  if (Insn.isa<spv::OpImageSampleExplicitLod>()) {
    if (!ResultIdMap.count(Insn.getOperand(0)))
      return FilterAction::Keep;

    // Replace sampled-image operand from dropped bitcast to its input operand.
    ReplacementInsn.assign(&Insn.getWord(0), &Insn.getWord(0) + Insn.size());
    ReplacementInsn[3] = ResultIdMap[Insn.getOperand(0)];
    return FilterAction::Replace;
  }

  return FilterAction::Keep;
}

// TODO/FIXME: 'const char *Bytes' --> 'const uint32_t *Words' as this function
// expects the Bytes to be aligned to 32-bits.
bool preprocessSPIRV(const char *Bytes, size_t NumBytes,
                     bool PreventNameDemangling, std::vector<uint32_t> &Dst) {
  assert(reinterpret_cast<uintptr_t>(Bytes) % sizeof(InstWord) == 0 &&
         "Expected SPIR-V word aligned input!");
  logTrace("preprocessSPIRV");

  auto *WordsBegin = reinterpret_cast<const InstWord *>(Bytes);
  auto *WordsPtr = WordsBegin;
  size_t NumWords = NumBytes / sizeof(InstWord);
  Dst.clear();
  Dst.reserve(NumWords);

  if (!parseHeader(WordsPtr, NumWords))
    return false; // Invalid SPIR-V binary.

  Dst.insert(Dst.end(), WordsBegin, WordsPtr); // Copy the header.
  logDebug("preprocessSPIRV: Added {} header words.", Dst.size());

  // Matches chipStar device library and SPIR-V translator symbols.
  auto CompilerMagicSymbol =
      std::regex(R"RE((__spirv_|__chip_|_Z\d*__chip_).*)RE");
  std::set<std::string_view> EntryPoints;
  std::unordered_set<InstWord> BuiltIns;
  std::unordered_map<InstWord, std::string_view> MissingDefs;
  IdMapT ResultIdMap;
  IdSetT SampledImgs;
  size_t InsnSize = 0;
  for (size_t I = 0; I < NumWords; I += InsnSize) {
    SPIRVinst Insn(WordsPtr + I);
    InsnSize = Insn.size();
    assert(InsnSize && "Invalid instruction size, will loop forever!");

    if (Insn.isEntryPoint())
      EntryPoints.insert(Insn.entryPointName());

    if (Insn.isExtension() && Insn.getExtension() == "SPV_KHR_linkonce_odr") {
      // Drop SPV_KHR_linkonce_odr and LinkOnceODR linkage attributes
      // (below) for improving portability. They appear for inline and
      // template functions. Dealing with fully linked device code the
      // attribute is no longer needed.
      logWarn("preprocessSPIRV: Dropped OpExtension SPV_KHR_linkonce_odr. Op: {}, Size: {} words", (int)Insn.getOpcode(), Insn.size());
      continue;
    }

    // A workaround for https://github.com/CHIP-SPV/chipStar/issues/48.
    //
    // Some Intel Compute Runtime versions fails to compile valid SPIR-V
    // modules correctly on OpenCL if there are OpEntryPoints and
    // functions or export linkage attributes by the same name.
    //
    // This workaround drops OpName instructions, whose string matches one of
    // the OpEntryPoint names, and all linkage attribute OpDecorations from the
    // binary we don't need to preserve. OpNames do not have semantical meaning
    // and we are not currently linking the SPIR-V modules with anything else.
    // if (Insn.isName() && EntryPoints.count(Insn.getName())) {
    //   logWarn("preprocessSPIRV: Dropped OpName for EntryPoint '{}'. Op: {}, TargetID: {}, Name: '{}'", Insn.getName(), (int)Insn.getOpcode(), Insn.getWord(1), Insn.getName());
    //   continue;
    // }

    if (Insn.isDecoration(spv::DecorationLinkageAttributes)) {
      auto LinkName = parseLinkageAttributeName(Insn);
      if (EntryPoints.count(LinkName)) {
        // logWarn("preprocessSPIRV: Dropped OpDecorate LinkageAttributes for EntryPoint '{}'. Op: {}, TargetID: {}, LinkName: '{}'", LinkName, (int)Insn.getOpcode(), Insn.getWord(1), LinkName);
        // continue;
      }
      if (parseLinkageAttributeType(Insn) == spv::LinkageTypeLinkOnceODR) {
        // Drop because the SPV_KHR_linkonce_odr is dropped in the above.
        logWarn("preprocessSPIRV: Dropped OpDecorate LinkageAttributes (LinkOnceODR). Op: {}, TargetID: {}, LinkName: '{}'", (int)Insn.getOpcode(), Insn.getWord(1), LinkName);
        continue;
      }
      if (parseLinkageAttributeType(Insn) == spv::LinkageTypeImport)
        // We are currently supposed to receive only fully linked
        // device code (from the user perspective). The user probably
        // forgot a definition.
        //
        // Issue warning unless it's a builtin, magic chipStar or
        // llvm-spirv symbol.
        if (!std::regex_match(LinkName.begin(), LinkName.end(),
                              CompilerMagicSymbol) &&
            !BuiltIns.count(Insn.getWord(1)))
          MissingDefs[Insn.getWord(1)] = LinkName;
    }

    if (Insn.isDecoration(spv::DecorationBuiltIn)) {
      BuiltIns.insert(Insn.getWord(1));
      MissingDefs.erase(Insn.getWord(1));
    }

#ifdef CHIP_MALI_GPU_WORKAROUNDS
    // (Old) Mali GPU drivers have issues consuming valid SPIR-V
    // modules with NoWrite and NoReadWrite parameter attributes so
    // drop them. Dropping these should not affect the module's
    // behavior.
    if (Insn.isDecoration(spv::DecorationFuncParamAttr)) {
      auto FnAttr = parseFunctionParameterAttribute(Insn);
      if (FnAttr == spv::FunctionParameterAttributeNoWrite ||
          FnAttr == spv::FunctionParameterAttributeNoReadWrite) {
        logWarn("preprocessSPIRV: Dropped OpDecorate FuncParamAttr (NoWrite/NoReadWrite). Op: {}, TargetID: {}, Attr: {}", (int)Insn.getOpcode(), Insn.getWord(1), (int)FnAttr);
        continue;
      }
    }
#endif

    std::vector<InstWord> TransformedInst;
    switch (workaroundLlvmSpirvIssue2008(Insn, TransformedInst, ResultIdMap,
                                         SampledImgs)) {
    default:
      assert(false && "Unknown instruction filter action!");
      // FALLTHROUGH
    case FilterAction::Error: // Assuming Error means keep for now or should be handled
        logError("preprocessSPIRV: Error action from workaroundLlvmSpirvIssue2008 for Insn Op: {}", (int)Insn.getOpcode());
        // Decide if to continue or treat as Keep. Current code falls through to Keep.
        // For safety, let's treat Error as Keep to match fallthrough.
    case FilterAction::Keep:
      break;
    case FilterAction::Drop:
      logWarn("preprocessSPIRV (workaroundLlvmSpirvIssue2008): Dropped Insn. Op: {}, ResultID: {}", (int)Insn.getOpcode(), Insn.hasResultID() ? (int)Insn.getResultID() : -1);
      continue;
    case FilterAction::Replace: {
      logDebug("preprocessSPIRV (workaroundLlvmSpirvIssue2008): Modifying Insn Op: {}, ResultID: {}. Will be replaced.", (int)Insn.getOpcode(), Insn.hasResultID() ? (int)Insn.getResultID() : -1);
      Dst.insert(Dst.end(), TransformedInst.begin(), TransformedInst.end());
      if (!TransformedInst.empty()) {
          SPIRVinst TempAddedInsn(TransformedInst.data());
          logDebug("preprocessSPIRV: Added (as replacement from workaround) Op: {}, ResultID: {}, Size: {} words", (int)TempAddedInsn.getOpcode(), TempAddedInsn.hasResultID() ? (int)TempAddedInsn.getResultID() : -1, TempAddedInsn.size());
      } else {
          logDebug("preprocessSPIRV: Added (as replacement from workaround) an empty instruction vector.");
      }
      continue;
    }
    }

    // see SPVRegister::PreventNameDemangling
    if (PreventNameDemangling && Insn.isEntryPoint()) {
      logDebug("preprocessSPIRV (PreventNameDemangling): Modifying OpEntryPoint Op: {}, ID: {}, Name: '{}'", (int)Insn.getOpcode(), Insn.entryPointID(), Insn.entryPointName());
      TransformedInst.assign(&Insn.getWord(0), &Insn.getWord(0) + Insn.size());
      char *Temp = reinterpret_cast<char *>(TransformedInst.data() + 3);
      std::swap(Temp[0], Temp[1]);
      Dst.insert(Dst.end(), TransformedInst.begin(), TransformedInst.end());
      if (!TransformedInst.empty()) {
          SPIRVinst TempAddedInsn(TransformedInst.data());
          logDebug("preprocessSPIRV: Added (as modification from PreventNameDemangling) Op: {}, ID: {}, Size: {} words", (int)TempAddedInsn.getOpcode(), TempAddedInsn.entryPointID(), TempAddedInsn.size());
      } else {
          logDebug("preprocessSPIRV: Added (as modification from PreventNameDemangling) an empty instruction vector.");
      }
      continue;
    }

    // logDebug("preprocessSPIRV: Added (copied) Insn. Op: {}, ResultID: {}, Size: {} words", (int)Insn.getOpcode(), (Insn.hasResultID() ? (int)Insn.getResultID() : -1), Insn.size());
    Dst.insert(Dst.end(), WordsPtr + I, WordsPtr + I + InsnSize);
  }

  for (auto &[Ignored, Name] : MissingDefs)
    logWarn("Missing definition for '{}'", Name);

  return true;
}

bool postprocessSPIRV(std::vector<uint32_t> &Input) {
  logTrace("postprocessSPIRV");

  std::vector<uint32_t> Result;
  Result.reserve(Input.size());

  // Copy the header.
  assert(Input.size() >= 5 && "SPIR-V header missing?");
  Result.insert(Result.end(), Input.begin(), std::next(Input.begin(), 5));

  // Processing in two pass for dealing with forward ID references.

  // Instructions and decorators with the result ID to be erased.
  std::set<InstWord> InstructionsToErase;

  // First pass.
  size_t InsnSize = 0;
  constexpr size_t PastHeader = 5;
  for (size_t I = PastHeader; I < Input.size(); I += InsnSize) {
    SPIRVinst Insn(&Input.at(I));
    InsnSize = Insn.size();
    assert(InsnSize && "Invalid instruction size, will loop forever!");

    if (Insn.isDecoration(spv::DecorationLinkageAttributes)) {
      const auto LinkName = parseLinkageAttributeName(Insn);

      // Remove chipStar runtime metadata variables which are
      // expressed as global-scope variables. This is accommodation
      // for mesa/rusticl that does not support them yet. Also, the
      // variables are essentially dead code for the driver.
      if (LinkName == "__chip_module_has_no_IGBAs" ||
          startsWith(LinkName, "__chip_spilled_args_"))
        InstructionsToErase.insert(Insn.getWord(1));
    }
  }

  // Second pass.
  for (size_t I = PastHeader; I < Input.size(); I += InsnSize) {
    SPIRVinst Insn(&Input.at(I));
    InsnSize = Insn.size();
    assert(InsnSize && "Invalid instruction size, will loop forever!");

    InstWord FwdRefID;
    if (Insn.isForwardReferencing(&FwdRefID) &&
        InstructionsToErase.count(FwdRefID)) {
      logTrace("filter out fwd ref id={}", FwdRefID);
      continue;
    }

    if (Insn.hasResultID() && InstructionsToErase.count(Insn.getResultID())) {
      logTrace("filter out inst id={}", Insn.getResultID());
      continue;
    }

    Result.insert(Result.end(), &Input.at(I), &Input.at(I) + InsnSize);
  }

  Input = std::move(Result);

  return true;
}

bool analyzeSPIRV(InstWord *Stream, size_t NumWords, SPVModuleInfo &Output) {
  SPIRVmodule Mod;
  if (!Mod.analyzeSPIRV(Stream, NumWords))
    return false;
  return Mod.fillModuleInfo(Output);
}
