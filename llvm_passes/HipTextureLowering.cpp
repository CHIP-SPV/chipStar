//===- HipTextureLowering.cpp ---------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A pass to lower HIP texture functions.
//
// (c) 2022 Henry Linjamäki / Parmance for Argonne National Laboratory
//===----------------------------------------------------------------------===//

#include "HipTextureLowering.h"

#include "LLVMSPIRV.h"
#include "../src/common.hh"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Utils/Cloning.h"

#if LLVM_VERSION_MAJOR >= 17
#include "SPIRVImageType.hh"
#endif

#define PASS_ID "hip-tex-lowering"
#define DEBUG_TYPE PASS_ID

using namespace llvm;

namespace {

namespace TexType {
enum {
  Unresolved = 0, // Must be convertible to false. Indicates that there are no
                  // users that depends on the contents of a texture object.
  Unknown = (1u << 0), // Indicates that there is a user that consumes a texture
                       // object but the texture type need is not known.
  Basic1D = (1u << 1),
  Basic2D = (1u << 2),

  // Mask for OpenCL supported texture types.
  OpenCLSupportedTypes = Basic1D | Basic2D,
};
}

// A set of texture types implemented as a bit vector.
using TexTypeSet = unsigned;

static bool isTextureFunctionCall(Instruction *I, TexTypeSet &Out) {
  if (auto *CI = dyn_cast<CallInst>(I)) {
    if (auto *F = CI->getCalledFunction()) {
      Out = StringSwitch<TexTypeSet>(F->getName())
                .StartsWith("_chip_tex1d", TexType::Basic1D)
                .StartsWith("_chip_tex2d", TexType::Basic2D)
                .Default(TexType::Unresolved);

      return Out != TexType::Unresolved;
    }
  }
  return false;
}

static bool isTextureFunctionCall(Instruction *I) {
  TexTypeSet Ignored;
  return isTextureFunctionCall(I, Ignored);
}

// Captures a set of texture object sources and texture function calls that uses
// them.
class TextureUseGroup {
  // Texture object sources that are directly passed in kernel arguments.
  std::set<Argument *> DirectTexSources_;

  // Indirect texture object sources - e.g. ones that are wrapped in aggregates.
  std::set<Value *> IndirectTexSources_;

  std::set<CallInst *> TexFunctionCalls_;

  std::set<User *> UnknownTexUsers_;

public:
  void recordDirectTexSource(Argument *V) {
    assert(!IndirectTexSources_.count(V));
    DirectTexSources_.insert(V);
  }
  void recordIndirectTexSource(Value *V) { IndirectTexSources_.insert(V); }
  void recordTexFunctionCall(CallInst *V) {
    assert(!IndirectTexSources_.count(V));
    assert(!UnknownTexUsers_.count(V));
    TexFunctionCalls_.insert(V);
  }
  void recordUnknownTexUser(User *V) {
    assert(!IndirectTexSources_.count(V));
    UnknownTexUsers_.insert(V);
  }

  void dump() const;

  const std::set<Argument *> &getDirectTexSources() const {
    return DirectTexSources_;
  }
  const std::set<CallInst *> &getTexFunctionCalls() const {
    return TexFunctionCalls_;
  }

  bool hasIndirectTexSources() const { return !IndirectTexSources_.empty(); }
  bool hasUnknownTexUsers() const { return !UnknownTexUsers_.empty(); }
  unsigned getNumTextureSources() const {
    return DirectTexSources_.size() + IndirectTexSources_.size();
  }
  unsigned getNumTextureCalls() const { return TexFunctionCalls_.size(); }

  bool hasSingleOpenCLSupportedTexType(TexTypeSet &Out) const {
    TexTypeSet TySet =
        UnknownTexUsers_.empty() ? TexType::Unresolved : TexType::Unknown;
    for (auto *TexCall : TexFunctionCalls_) {
      TexTypeSet CallTySet;
      if (isTextureFunctionCall(TexCall, CallTySet))
        TySet |= CallTySet;
    }
    if (isPowerOf2_64(TySet) && (TexType::OpenCLSupportedTypes & TySet)) {
      Out = TySet;
      return true;
    }
    return false;
  }
};

void TextureUseGroup::dump() const {
#define DOIT(_container, _desc)                                                \
  do {                                                                         \
    dbgs() << _desc << ":" << (_container.empty() ? " none" : "") << "\n";     \
    for (auto *Elt : _container) {                                             \
      dbgs() << "- ";                                                          \
      Elt->dump();                                                             \
    }                                                                          \
  } while (0)

  DOIT(DirectTexSources_, "Direct tex sources");
  DOIT(IndirectTexSources_, "Indirect tex sources");
  DOIT(TexFunctionCalls_, "Tex function calls");
  DOIT(UnknownTexUsers_, "Unknown tex users");
#undef DOIT
}

using TextureUseGroupList = std::vector<TextureUseGroup>;
using TextureUseGroupMap = std::map<Value *, TextureUseGroup *>;

/// Consider 'V' as a texture object. This analysis method searches its users
/// and sources and records them in the 'TexUseGroup'.
///
/// 'V' serves as a starting point for the searching and it must not be a
/// constant. The 'TexUseGroupMap' keeps track for already covered instructions
/// (per kernel as its intention). Callers should not populate this map. The
/// 'VisitEdge' is private to this method. Callers should pass its default
/// value.
static void analyze(Value *V, TextureUseGroup &TexUseGroup,
                    // For tracking already covered instructions (per kernel).
                    // Callers should not populate this map.
                    TextureUseGroupMap &TexUseGroupMap,
                    // A edge which was followed to the 'V'.
                    // A analysis start point if nullptr.
                    Use *VisitEdge = nullptr) {

  auto VisitDefsAndOrUses = [&](Value *Start, bool VisitUses, bool VisitDefs,
                                ArrayRef<unsigned> DefOpdIdxs) -> void {
    if (VisitUses)
      for (auto &Use : Start->uses())
        analyze(Use.getUser(), TexUseGroup, TexUseGroupMap, &Use);

    if (VisitDefs) {
      if (auto *TheUser = dyn_cast<User>(Start)) {
        for (auto DefOpdIdx : DefOpdIdxs) {
          auto &Use = TheUser->getOperandUse(DefOpdIdx);
          analyze(Use.get(), TexUseGroup, TexUseGroupMap, &Use);
        }
      }
    }
  };

  auto VisitUses = [&](Value *Start) -> void {
    VisitDefsAndOrUses(Start, true, false, 0);
  };
  auto VisitDef = [&](Value *Start, unsigned DefOpdIdx) -> void {
    VisitDefsAndOrUses(Start, false, true, DefOpdIdx);
  };
  auto VisitDefsAndUses = [&](Value *Start,
                              ArrayRef<unsigned> DefOpdIdxs) -> void {
    VisitDefsAndOrUses(Start, true, true, DefOpdIdxs);
  };

  if (TexUseGroupMap.count(V)) { // Already visited and processed?
    // Check this analysis run does not overlap with a previous one.
    // FIXME/TODO: Only accepted overlap is nodes that are classified as unknown
    //             source and texture user.
    assert(TexUseGroupMap[V] == &TexUseGroup);
    return;
  }

  if (isa<Constant>(V)) { // Constants are ignored.
    LLVM_DEBUG(dbgs() << " Ignore:" << *V << "\n");
    // Starting point of the analysis must not be a constant.
    assert(VisitEdge);
    return;
  }

  LLVM_DEBUG(dbgs() << "Analyze:" << *V << "\n");
  TexUseGroupMap[V] = &TexUseGroup;

  if (auto *Arg = dyn_cast<Argument>(V)) {
    TexUseGroup.recordDirectTexSource(Arg);
    VisitUses(V);
    return;
  }

  auto *I = cast<Instruction>(V);

  if (isTextureFunctionCall(I)) {
    LLVM_DEBUG(dbgs() << "  A tex function call.\n");
    TexUseGroup.recordTexFunctionCall(cast<CallInst>(I));
    VisitDef(I, 0);
    return;
  }

  switch (I->getOpcode()) {
  default:
    break;
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast:
    VisitDefsAndUses(I, {0});
    return;
  case Instruction::Select:
    VisitDefsAndUses(I, {1, 2});
    return;
  }

  // TODO: Look through loads and stores of local variables? These cases would
  // occur on -O0 (unless SROA pass is forced).

  // A texture object is received from or passed to an unrecognized
  // instruction.

  // TODO: handle VisitEdge == nullptr case?
  assert(VisitEdge);
  if (VisitEdge && VisitEdge->getUser() == I) {
    LLVM_DEBUG(dbgs() << "  Unknown tex user\n");
    TexUseGroup.recordUnknownTexUser(I);
  } else {
    LLVM_DEBUG(dbgs() << "  Indirect or unknown tex source\n");
    TexUseGroup.recordIndirectTexSource(I);
  }
}

static TextureUseGroupList analyzeTextureObjectUses(Function &F) {
  SmallVector<Value *, 4> StartPoints;
  for (auto &BB : F)
    for (auto &I : BB) {
      if (!isTextureFunctionCall(&I))
        continue;
      // TODO: assert on regular function call (unsupported).
      LLVM_DEBUG(dbgs() << "Analysis start point: " << I << "\n");
      StartPoints.push_back(&I);
    }

  TextureUseGroupList Result;
  TextureUseGroupMap TexUseGroupMap;
  for (auto StartPoint : StartPoints) {
    if (TexUseGroupMap.count(StartPoint))
      continue;
    Result.emplace_back();
    LLVM_DEBUG(dbgs() << "\n");
    analyze(StartPoint, Result.back(), TexUseGroupMap);
  }

  return Result;
}

#if LLVM_VERSION_MAJOR < 17
// Create a pointer type to to named opaque struct.
static Type *getPointerTypeToOpaqueStruct(LLVMContext &C, StringRef Name,
                                          unsigned AddrSpace = 0) {
  Type *Ty = StructType::getTypeByName(C, Name);
  if (!Ty)
    Ty = StructType::create(C, Name);
  return Ty->getPointerTo(AddrSpace);
}
#endif

static Type *getSamplerType(LLVMContext &C) {
#if LLVM_VERSION_MAJOR < 17
  return getPointerTypeToOpaqueStruct(C, "opencl.sampler_t", OCL_SAMPLER_AS);
#else
  return TargetExtType::get(C, "spirv.Sampler");
#endif
}



static Type *getImageType(LLVMContext &C, TexTypeSet TexTySet) {
  switch (TexTySet) {
  default:
  case TexType::Unresolved:
    llvm_unreachable("Expected single image type.");
    return nullptr;
  case TexType::Basic1D:
  {
#if LLVM_VERSION_MAJOR < 17
    return getPointerTypeToOpaqueStruct(C, "opencl.image1d_ro_t", OCL_IMAGE_AS);
#else
    return getSPIRVImageType(C, "spirv.Image", "image1d", AQ_ro);
#endif
  }
  case TexType::Basic2D:
  {
#if LLVM_VERSION_MAJOR < 17
    return getPointerTypeToOpaqueStruct(C, "opencl.image2d_ro_t", OCL_IMAGE_AS);
#else
    return getSPIRVImageType(C, "spirv.Image", "image2d", AQ_ro);
#endif
  }

  }
}

// Create a temporary value definition (an instruction) which is intended to be
// replaced later with an actual definition.
static Instruction *getPlaceholder(Type *Ty, Function *F) {
  // Use the freeze instruction from a poison value as a temporary definition.
  auto *PV = PoisonValue::get(Ty);
  return new FreezeInst(PV, "placeholder",
                        F->getEntryBlock().getFirstNonPHIOrDbg());
}

static void lowerTextureObjectUses(Function *F,
                                   const TextureUseGroupList &TexUseGroups) {
  if (TexUseGroups.empty())
    return;

  // The implementation expects that the kernels are not being called by other
  // kernels (should not happen as CUDA/HIP kernels may not call other kernels).
  assert(F->hasNUses(0));

  auto *M = F->getParent();
  auto &C = M->getContext();
  std::map<Argument *, std::vector<Value *>> ArgsToExpand;
  SmallVector<Instruction *, 4> EraseList;
  for (const auto &TexUseGroup : TexUseGroups) {
    TexTypeSet TexTy;
    if (!TexUseGroup.hasIndirectTexSources() &&
        !TexUseGroup.hasUnknownTexUsers() &&
        TexUseGroup.getNumTextureSources() == 1 &&
        TexUseGroup.hasSingleOpenCLSupportedTexType(TexTy)) {
      // A basic texture use case.
      LLVM_DEBUG(dbgs() << "Lower a basic texture use case\n");

      // Expand hipTextureObject argument to OpenCL image and sampler
      // arguments.
      Argument *Arg = *TexUseGroup.getDirectTexSources().begin();
      Type *ImgArgTy = getImageType(C, TexTy);
      Type *SplArgTy = getSamplerType(C);
      Value *ImgArgPlaceholder = getPlaceholder(ImgArgTy, F);
      Value *SplArgPlaceholder = getPlaceholder(SplArgTy, F);
      ArgsToExpand[Arg].push_back(ImgArgPlaceholder);
      ArgsToExpand[Arg].push_back(SplArgPlaceholder);

      // Replace texture call with a call to the actual implementation.
      for (auto *CI : TexUseGroup.getTexFunctionCalls()) {
        auto *ImplF = M->getFunction(
            (CI->getCalledFunction()->getName() + "_impl").str());
        assert(ImplF);
        // Create call to the actual implementation.
        SmallVector<Value *, 4> CallArgs{ImgArgPlaceholder, SplArgPlaceholder};
        // Copy the rest from the old call past the texture object argument.
        for (unsigned I = 1, E = CI->arg_size(); I != E; I++)
          CallArgs.push_back(CI->getArgOperand(I));
        auto *NewCI =
            CallInst::Create(ImplF->getFunctionType(), ImplF, CallArgs, "", CI);
        // Calling convention is not inherited from the callee.
        NewCI->setCallingConv(CallingConv::SPIR_FUNC);

        CI->replaceAllUsesWith(NewCI);
        EraseList.push_back(CI);
      }
      continue;
    }

    // If we reach here it means:
    //
    // 1) the texture use case, which could be lowered to use OpenCL
    //    image read functions, is not implemented or
    //
    // 2) the texture use case can not be implemented using OpenCL image read
    //    functions. In this case the texture function calls should be emulated
    //    (which is not implemented).
    llvm_unreachable("Don't know how to lower this texture use case.");
  }

  for (auto *ToErase : EraseList)
    ToErase->eraseFromParent();

  // Clone the kernel if needed for changing its parameter list.
  if (ArgsToExpand.empty())
    return;

  // Prepare new kernel function and its parameters to clone the current kernel
  // into.

  // Create new argument list.
  SmallVector<Type *, 8> NewArgTys;
  for (auto &Arg : F->args()) {
    if (ArgsToExpand.count(&Arg))
      for (auto *V : ArgsToExpand[&Arg])
        NewArgTys.push_back(V->getType());
    else
      NewArgTys.push_back(Arg.getType());
  }

  // Create new kernel function.
  auto *NewFnTy =
      FunctionType::get(F->getReturnType(), NewArgTys, F->isVarArg());
  auto *NewF =
      Function::Create(NewFnTy, F->getLinkage(), F->getAddressSpace(), "", M);
  NewF->copyAttributesFrom(F);

  // Prepare argument mapping for the function cloning.
  ValueToValueMapTy VMap;
  ValueToValueMapTy PlaceholderToArg;
  auto NewFArgIt = NewF->arg_begin();
  for (auto &Arg : F->args()) {
    if (ArgsToExpand.count(&Arg)) {
      for (auto *ArgPlaceholder : ArgsToExpand[&Arg])
        PlaceholderToArg[ArgPlaceholder] = &(*NewFArgIt++);
      // The old argument and code depending on it will be unused.
      VMap[&Arg] = PoisonValue::get(Arg.getType());
    } else
      VMap[&Arg] = &(*NewFArgIt++);
  }

  // Clone the function.
  SmallVector<ReturnInst *, 8> Ignored;
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Ignored, "");

  // Map argument placeholders to the actual arguments.
  for (auto I : PlaceholderToArg) {
    const Value *From = I.first;
    Value *To = I.second;
    Value *ClonedV = VMap[From];
    ClonedV->replaceAllUsesWith(To);
  }

  // Remove the old function, reclaim the original function name.
  std::string OrigName = F->getName().str();
  F->eraseFromParent();
  NewF->setName(OrigName);
}

static bool lowerTextureFunctions(Module &M) {
  bool Changed = false;

  SmallPtrSet<Function *, 16> Worklist;
  for (auto &F : M) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      Worklist.insert(&F);
      continue;
    }

#ifndef NDEBUG
    // This pass does not work on kernels where the called texture function are
    // in other function (for now). The device code should be fully inlined.
    for (auto &BB : F)
      for (auto &I : BB)
        if (isTextureFunctionCall(&I)) {
          assert(false && "Unsupported texture function use.");
          return Changed;
        }
#endif
  }

  for (auto *F : Worklist) {
    auto TexUseGroups = analyzeTextureObjectUses(*F);
    if (TexUseGroups.empty()) {
      LLVM_DEBUG(dbgs() << "No texture functions in '" << F->getName()
                        << "\n";);
      continue;
    }

    LLVM_DEBUG(unsigned Id = 0;
               dbgs() << "Texture use groups in " << F->getName() << ":\n";
               for (auto &TexUseGroup
                    : TexUseGroups) {
                 dbgs() << "Group " << Id++ << ":\n";
                 TexUseGroup.dump();
               });

    lowerTextureObjectUses(F, TexUseGroups);
    Changed = true;
  }

  return Changed;
}

} // namespace

PreservedAnalyses HipTextureLoweringPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  return lowerTextureFunctions(M) ? PreservedAnalyses::none()
                                  : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_ID, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_ID) {
                    MPM.addPass(HipTextureLoweringPass());
                    return true;
                  }
                  return false;
                });
          }};
}
