//===- HipKernelArgSpiller.cpp --------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Reduces the size of large kernel parameter lists by "spilling" them.
//
// In CUDA (and HIP too?) kernel parameter lists are limited to 4KB [1] but
// chipStar's backends may have more stricter limit. We support large kernel
// parameter lists on such backends by passing some of the arguments via
// intermediate, temporary device buffer.
//
// This pass modifies some of the kernel argument by converting them into
// buffer pointers. For example (through CUDA analogy):
//
//   __global__ void aKernel(BigObj A, BigObj B, ...) { ... }
//
//   -->
//
//   __device__ void aKernel_original(BigObj A, BigObj B, ...) { ... }
//   __global__ void aKernel(BigObj *A, BigObj *B, ...) {
//     aKernel_original(*A, *B, ...);
//   }
//
// The chipStar runtime is let to know about the spilled arguments by annotating
// their position and original size of the argument in a global magic array:
//
//    uint16_t __chip_spilled_args_<kernel-name>[] = {
//      <argument position>, <argument-size>,
//      <argument position>, <argument-size>,
//      ...
//    };
//
// One annotation per kernel is emitted at most. The absence of this variable
// means no argument has been spilled.
//
// [1]: CUDA C++ Programming Guide 14.5.9.3. Function Parameters
//
// Copyright (c) 2023 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipKernelArgSpiller.h"

#include "LLVMSPIRV.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

#include <optional>

#define PASS_NAME "hip-spill-kernel-args"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

namespace {

// Maximum kernel parameter list size.
//
// Start with OpenCL's minimum mandated kernel parameter list size for all
// devices in full profile (CL_DEVICE_MAX_ARGUMENT_SIZE).
constexpr size_t MAX_KERNEL_PARAM_LIST_SIZE = 1024;

/// Return true if the type is a "special" one in SPIR-V.
///
/// For example, the type maps to an image or sampler type. IOW, the type maps
/// to other than an integer, a floating-point, a vector or an aggregate type.
/// Generally, the special types don't have size and/or they are not spillable.
bool isSpecial(const Argument &Arg) {
#if LLVM_VERSION_MAJOR >= 17
  Type *Ty = Arg.getType();
  return Ty->isTargetExtTy();
#elif LLVM_VERSION_MAJOR == 16
  // Can't detect image and sampler arguments/types straightforwardly in LLVM 16
  // because
  //
  // * named opaque struct types are gone in the opaque pointer world and
  //
  // * the alternative way to express them
  //   (https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/1880) didn't
  //   make it to the llvm-spirv branch "llvm_release_160".
  //
  // So until LLVM 17 we consider all pointer arguments as non-special. We won't
  // end up with spilling image and sampler arguments with the current
  // implementation as they don't have byval attribute and non-byval pointers
  // are not profitable to spill.
  return false;
#else
  // Special types are encoded as pointers to specially named structures (until
  // typed pointers leaves the LLVM)
  Type *Ty = Arg.getType();
  auto *PtrTy = dyn_cast<PointerType>(Ty);
  if (!PtrTy)
    return false;

  auto *STy = dyn_cast<StructType>(PtrTy->getNonOpaquePointerElementType());
  if (!STy || !STy->hasName())
    return false;

  // Only consider OpenCL and SPIR-V types - leave user defined opaque
  // structures alone (they are just plain pointers).
  StringRef Name = STy->getName();
  return (Name.startswith("opencl.") || Name.startswith("__spirv_"));
#endif
}

static size_t getArgumentSize(const Argument &Arg) {
  assert(!isSpecial(Arg) &&
         "Can't determine the size of a special SPIR-V type!");

  auto &DL = Arg.getParent()->getParent()->getDataLayout();
  if (Arg.hasByValAttr()) // Byval pointees travel in the kernel arg buffer.
    return DL.getTypeStoreSize(Arg.getParamByValType());
  return DL.getTypeStoreSize(Arg.getType());
}

static bool canSpill(const Argument &Arg) {
  if (isSpecial(Arg))
    return false;

  auto *ArgTy = Arg.getType();
  if (isa<PointerType>(ArgTy))
    if (ArgTy->getPointerAddressSpace() == SPIRV_WORKGROUP_AS ||
        ArgTy->getPointerAddressSpace() == SPIRV_UNIFORMCONSTANT_AS)
      return false;

  return true;
}

using ArgSet = SmallPtrSet<const Argument *, 16>;

static ArgSet getSpillPlan(Function *F) {
  size_t ArgumentsSize = 0;
  for (const Argument &Arg : F->args()) {
    if (isSpecial(Arg)) {
      // For starters, we are not currently handling kernels with image and
      // sampler and other special object arguments. It's not clear what their
      // sizes really are.
      LLVM_DEBUG(dbgs() << "  Bail out: Kernel has a special arg: " << Arg
                        << ".\n");
      return ArgSet();
    }
    ArgumentsSize += getArgumentSize(Arg);
  }

  if (ArgumentsSize <= MAX_KERNEL_PARAM_LIST_SIZE)
    return ArgSet();

  // TODO: give a warning if argument list size exceeds CUDA's 4KB limit.

  LLVM_DEBUG(dbgs() << "  Arg list (" << ArgumentsSize
                    << " B) exceeds the arg buffer limit of "
                    << MAX_KERNEL_PARAM_LIST_SIZE << " B.\n");

  // Arguments are spilled into a runtime managed buffer.
  auto &DL = F->getParent()->getDataLayout();
  auto PointerSize = DL.getPointerSize(SPIRV_CROSSWORKGROUP_AS);
  ArgSet ArgsToSpill;

  // TODO: Optimization. Prefer spilling largest arguments first so we have
  //       fewest amount of copy instances.  Additionally, if we need to spill X
  //       bytes to meet the limit and there are multiple arguments larger than
  //       that, pick the smallest one to minimize amount of bytes to be
  //       copied.

  // TODO: Optimization. Prefer spilling readonly byval arguments first, which
  //       are never modified. This gives opportunities to omit local copies.

  for (const auto &Arg : F->args()) {
    if (!canSpill(Arg)) {
      LLVM_DEBUG(dbgs() << "  Arg " << Arg.getArgNo() << ": Can't spill " << Arg
                        << ".\n");
      continue;
    }
    auto ParamSize = getArgumentSize(Arg);
    if (ParamSize <= PointerSize)
      continue; // No parameter size reduction when spilled.

    auto Reduction = ParamSize - PointerSize;
    LLVM_DEBUG(dbgs() << "  Arg " << Arg.getArgNo() << ": Spill. Reduction: -"
                      << Reduction << " bytes.\n");
    ArgumentsSize -= Reduction;
    ArgsToSpill.insert(&Arg);
    if (ArgumentsSize <= MAX_KERNEL_PARAM_LIST_SIZE)
      break;
  }

  LLVM_DEBUG(dbgs() << "  Arg list after spilling: " << ArgumentsSize
                    << " B\n");

  if (ArgumentsSize > MAX_KERNEL_PARAM_LIST_SIZE) {
    LLVM_DEBUG(dbgs() << "  Bail out: arg list is still too large.\n");
    return ArgSet();
  }

  return ArgsToSpill;
}

/// Get type presentation for the argument. Always returns a pointer type.
static Type *getSpillType(const Argument &Arg) {
  auto *ArgTy = Arg.hasByValAttr() ? Arg.getParamByValType() : Arg.getType();
  return ArgTy->getPointerTo(SPIRV_CROSSWORKGROUP_AS);
}

/// Create an alloca placed in function's entry block.
static AllocaInst *createEntryAlloca(llvm::IRBuilder<> &B, Type *Ty) {
  auto SavedIP = B.saveIP();
  B.SetInsertPoint(
      B.GetInsertBlock()->getParent()->getEntryBlock().getFirstNonPHIOrDbg());
  auto *AI = B.CreateAlloca(Ty);
  B.restoreIP(SavedIP);
  return AI;
}

/// Annotate the spilled arguments for the runtime.
static void annotateSpilledArgs(Function *F, ArgSet ArgsToSpill) {
  // Store annotations in 32-bit array where lower 16-bit carriers argument
  // index of the spilled argument and upper 16-bits carry the size of the
  // argument.

  SmallVector<uint32_t> Annotations;
  for (const auto *Arg : ArgsToSpill) {
    assert(Arg->getArgNo() >> 16u == 0 && "Doesn't fit in 16-bit field!");
    assert(getArgumentSize(*Arg) >> 16u == 0 && "Doesn't fit in 16-bit field!");
    uint32_t Anno = (getArgumentSize(*Arg) << 16) | (0xffff & Arg->getArgNo());
    Annotations.emplace_back(Anno);
  }
  auto Name = Twine("__chip_spilled_args_") + F->getName();
  auto *GVInit = ConstantDataArray::get(F->getContext(), Annotations);
  auto *GV = new GlobalVariable(
      *F->getParent(), GVInit->getType(), true,
      // Mark the GV as external for keeping it alive at least until the
      // chipStar runtime reads it.
      GlobalValue::ExternalLinkage, GVInit, Name, nullptr,
      GlobalValue::NotThreadLocal /* Default value*/,
      // Global-scope variables may not have Function storage class.
      // TODO: use private storage class?
      SPIRV_CROSSWORKGROUP_AS);
  LLVM_DEBUG(dbgs() << "Annotated spilled args: " << *GV << "\n");
}

static bool spillKernelArgs(Function *F) {
  assert(F->getCallingConv() == CallingConv::SPIR_KERNEL);

  LLVM_DEBUG(dbgs() << "Visit kernel: " << F->getName() << ".\n");

  ArgSet ArgsToSpill = getSpillPlan(F);
  if (ArgsToSpill.empty())
    return false;

  // Implement the spill plan.

  // Create new kernel which replaces the current one. Convert the current
  // kernel into a function.
  SmallVector<Type *> NewArgTys;
  ValueToValueMapTy ArgMap;
  for (auto &Arg : F->args()) {
    NewArgTys.push_back(ArgsToSpill.count(&Arg) ? getSpillType(Arg)
                                                : Arg.getType());
  }
  auto *NewFnTy =
      FunctionType::get(F->getReturnType(), NewArgTys, F->isVarArg());
  auto *NewF = Function::Create(NewFnTy, F->getLinkage(), F->getAddressSpace(),
                                "", F->getParent());
  NewF->copyAttributesFrom(F);
  NewF->takeName(F);

  // Convert the original kernel into a regular function.
  F->setName(NewF->getName() + ".original_kernel");
  F->setCallingConv(CallingConv::SPIR_FUNC);
  F->setLinkage(GlobalValue::InternalLinkage);

  // Prepare IR builder.
  IRBuilder<> B(BasicBlock::Create(F->getContext(), "entry", NewF));
  auto *RI = B.CreateRetVoid();
  B.SetInsertPoint(RI);

  // Add "reload" code for the spilled arguments.
  const auto &DL = F->getParent()->getDataLayout();
  SmallVector<Value *> CallArgs;
  for (auto &OrigArg : F->args()) {
    auto *NewArg = NewF->getArg(OrigArg.getArgNo());
    if (!ArgsToSpill.count(&OrigArg)) {
      CallArgs.push_back(NewArg);
      continue;
    }

    // Emit copy from the spill buffer into a private variable to preserve
    // private ownership of the original pass-by-value argument.

    if (NewArg->hasByValAttr())
      NewArg->removeAttr(Attribute::ByVal);

    // TODO: optimization. Skip local copy emission if a byval argument is never
    //       modified (has readonly attribute).

#if LLVM_VERSION_MAJOR < 16
    auto *AllocaTy = NewArg->getType()->getNonOpaquePointerElementType();
#else
    auto *AllocaTy = OrigArg.hasByValAttr() ? OrigArg.getParamByValType()
                                            : OrigArg.getType();
#endif
    auto SrcAlign = DL.getABITypeAlign(AllocaTy);
    auto *LocalCopy = createEntryAlloca(B, AllocaTy);
    auto AllocSizeInBitsOpt = LocalCopy->getAllocationSizeInBits(DL);
    assert(AllocSizeInBitsOpt);
#if LLVM_VERSION_MAJOR > 17
    size_t AllocSizeInBits = AllocSizeInBitsOpt->getFixedValue();
#else
    size_t AllocSizeInBits = AllocSizeInBitsOpt->getFixedSize();
#endif
    assert(AllocSizeInBits % 8u == 0);
    size_t AllocSize = AllocSizeInBits / 8u;
    B.CreateMemCpy(LocalCopy, LocalCopy->getAlign(), NewArg, SrcAlign,
                   AllocSize);
    CallArgs.push_back(LocalCopy);
  }

  B.CreateCall(F, CallArgs); // Call the original kernel.

  annotateSpilledArgs(NewF, ArgsToSpill);

  return true;
}

static bool spillKernelArgs(Module &M) {
  bool Changed = false;

  SmallVector<Function *> WorkList;
  for (auto &F : M)
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
      WorkList.push_back(&F);

  for (auto *F : WorkList)
    Changed |= spillKernelArgs(F);

  return Changed;
}

} // namespace

PreservedAnalyses HipKernelArgSpillerPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {

  return spillKernelArgs(M) ? PreservedAnalyses::none()
                            : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_NAME, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_NAME) {
                    FPM.addPass(HipKernelArgSpillerPass());
                    return true;
                  }
                  return false;
                });
          }};
}
