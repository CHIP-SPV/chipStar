//===- HipLowerRoundIntrinsics.cpp ----------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Expands the type-crossing rounding intrinsics -- llvm.lround, llvm.llround,
// llvm.lrint and llvm.llrint -- which take a floating point value and return an
// integer. OpenCL.std has no instruction for that shape: every one of its
// rounding instructions (ceil, floor, rint, round, trunc) is float to float, so
// these four need a two instruction expansion rather than a 1:1 ExtInst. A
// producer that does not do that expansion itself rejects the module:
//
//   InvalidFunctionCall: Unexpected llvm intrinsic: llvm.llround.i64.f32
//
// and, with the in-tree SPIR-V backend:
//
//   LLVM ERROR: unable to legalize instruction: %8:iid(s64) = G_INTRINSIC_LRINT
//
// The failure surfaces at hipspv-link with no source location at all, and it
// cannot be headed off in the headers: libstdc++ declares these for float as
// constexpr, which HIP turns into implicitly __host__ __device__ functions, so
// they win overload resolution over anything chipStar could add (a plain
// __device__ overload with the same signature is rejected outright).
//
// So expand them here: round in floating point with llvm.round / llvm.rint,
// which both producers map to the OpenCL round and rint ExtInsts, then convert
// to the integer result. Calling the devicelib entry points instead does not
// work, because devicelib is linked in before the post-link passes run and only
// the symbols already referenced at that point are pulled in.
//
// All four are handled, not just the ones that happen to be untranslatable
// today, so that the set is defined by the shape of the intrinsic rather than
// by which producer version is in use. Of the four, only lround is handled by
// both producers as things stand, and both lower it to exactly this expansion
// (OpExtInst round, then OpConvertFToS), so covering it changes no emitted
// code. Taking the whole set also means the pass can be deleted in one go once
// the producers cover all of it.
//
// llvm.ldexp arrives the same way and dies the same way:
//
//   InvalidFunctionCall: Unexpected llvm intrinsic: llvm.ldexp.f64.i32
//
// There is no equivalent "expand into another intrinsic" trick for it -- the
// multiply-by-a-power-of-two rewrite loses subnormals and overflows -- but
// OpenCL has ldexp natively, so emit a call to the OpenCL builtin instead. That
// is not the same thing as calling devicelib: _Z5ldexpfi and _Z5ldexpdi stay
// *undefined* in the module (devicelib itself leaves them undefined, see
// bitcode/devicelib.cl) and llvm-spirv turns the call into the OpenCL ExtInst
// Ldexp, so nothing has to be linked in after the fact. HipPrintf.cpp declares
// the OpenCL printf the same way.
//
// (c) 2026 chipStar developers
//===----------------------------------------------------------------------===//

#include "HipLowerRoundIntrinsics.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include "PassPluginCompat.h"

#include <cstdint>

#define PASS_NAME "hip-lower-round-intrinsics"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;

/// The floating point rounding intrinsic a type-crossing rounder expands to, or
/// not_intrinsic when \p ID is something else.
static Intrinsic::ID getRoundingIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::lround:
  case Intrinsic::llround:
    // lround and llround round halfway cases away from zero, which is
    // llvm.round.
    return Intrinsic::round;
  case Intrinsic::lrint:
  case Intrinsic::llrint:
    // lrint and llrint follow the current rounding mode, which is llvm.rint.
    return Intrinsic::rint;
  default:
    return Intrinsic::not_intrinsic;
  }
}

static bool lowerRoundIntrinsics(Module &M) {
  SmallVector<CallInst *, 8> WorkList;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *Call = dyn_cast<CallInst>(&I))
          if (Function *Callee = Call->getCalledFunction())
            if (Call->arg_size() == 1 &&
                Call->getArgOperand(0)->getType()->isFloatingPointTy() &&
                getRoundingIntrinsic(Callee->getIntrinsicID()) !=
                    Intrinsic::not_intrinsic)
              WorkList.push_back(Call);

  for (auto *Call : WorkList) {
    Value *Arg = Call->getArgOperand(0);
    Intrinsic::ID Rounding =
        getRoundingIntrinsic(Call->getCalledFunction()->getIntrinsicID());

    IRBuilder<> Builder(Call);
    // Round in floating point, which llvm-spirv maps to the OpenCL round and
    // rint ExtInsts, then convert. The value is already integral at that point,
    // so the truncating conversion is exact.
    Value *Rounded = Builder.CreateUnaryIntrinsic(Rounding, Arg);
    Value *Result = Builder.CreateFPToSI(Rounded, Call->getType());

    Call->replaceAllUsesWith(Result);
    Call->eraseFromParent();
  }

  return !WorkList.empty();
}

/// The OpenCL C mangled name of ldexp for the scalar floating point type \p Ty,
/// or an empty string when OpenCL has no ldexp taking it. OpenCL declares
/// ldexp only over the half, float and double gentypes; those are also the only
/// floating point types that reach a SPIR-V device module, so anything else
/// (x86_fp80, fp128, bfloat) is deliberately left for the translator to reject
/// rather than silently mistranslated.
static StringRef getOpenCLLdexpName(Type *Ty) {
  if (Ty->isHalfTy())
    return "_Z5ldexpDhi";
  if (Ty->isFloatTy())
    return "_Z5ldexpfi";
  if (Ty->isDoubleTy())
    return "_Z5ldexpdi";
  return "";
}

/// Narrow \p Exp to the i32 that the OpenCL ldexp takes. llvm.ldexp accepts any
/// integer width and reads it as signed; clang only ever emits i32 here, but
/// InstCombine and SimplifyLibCalls also synthesise the intrinsic (from
/// exp2(sitofp x) and pow(2, sitofp x)) with whatever width the source integer
/// had. Clamping before the truncation keeps the wide case exact: every type
/// OpenCL's ldexp accepts has already saturated to zero or infinity long before
/// |exp| reaches 2^31, so a clamped exponent gives the same result as the
/// original one. A plain truncation would not: it turns 2^32 into 0.
static Value *coerceLdexpExponent(IRBuilder<> &Builder, Value *Exp) {
  auto *ExpTy = cast<IntegerType>(Exp->getType());
  if (ExpTy->getBitWidth() == 32)
    return Exp;
  if (ExpTy->getBitWidth() < 32)
    return Builder.CreateSExt(Exp, Builder.getInt32Ty());

  Value *Lo = ConstantInt::getSigned(ExpTy, INT32_MIN);
  Value *Hi = ConstantInt::getSigned(ExpTy, INT32_MAX);
  Value *Clamped =
      Builder.CreateSelect(Builder.CreateICmpSLT(Exp, Lo), Lo, Exp);
  Clamped = Builder.CreateSelect(Builder.CreateICmpSGT(Clamped, Hi), Hi,
                                 Clamped);
  return Builder.CreateTrunc(Clamped, Builder.getInt32Ty());
}

/// Emit a call to the OpenCL ldexp builtin for the scalar value \p X and the
/// i32 exponent \p Exp. The callee is left undefined on purpose: llvm-spirv
/// recognises the mangled name and emits the OpenCL ExtInst Ldexp for it.
static Value *emitOpenCLLdexp(IRBuilder<> &Builder, Module &M, Value *X,
                              Value *Exp) {
  Type *Ty = X->getType();
  FunctionCallee Callee = M.getOrInsertFunction(getOpenCLLdexpName(Ty), Ty, Ty,
                                                Builder.getInt32Ty());
  if (auto *F = dyn_cast<Function>(Callee.getCallee())) {
    // SPIR_FUNC is what marks this as a device side call for the translator,
    // and InternalizePass in the chipStar pipeline keys off it to keep the
    // declaration alive.
    F->setCallingConv(CallingConv::SPIR_FUNC);
    F->setVisibility(GlobalValue::HiddenVisibility);
    F->setDoesNotAccessMemory();
    F->setDoesNotThrow();
  }
  CallInst *Call = Builder.CreateCall(Callee, {X, Exp});
  Call->setCallingConv(CallingConv::SPIR_FUNC);
  return Call;
}

/// True when \p Call is an llvm.ldexp this pass knows how to rewrite.
static bool isLowerableLdexp(CallInst *Call) {
  Function *Callee = Call->getCalledFunction();
  if (!Callee || Callee->getIntrinsicID() != Intrinsic::ldexp ||
      Call->arg_size() != 2)
    return false;

  Type *Ty = Call->getType();
  // Scalable vectors cannot occur on a SPIR-V device module and cannot be
  // scalarised, so leave them be.
  if (isa<ScalableVectorType>(Ty))
    return false;
  if (!getOpenCLLdexpName(Ty->getScalarType()).empty() &&
      Call->getArgOperand(1)->getType()->isIntOrIntVectorTy())
    return true;
  return false;
}

static bool lowerLdexpIntrinsics(Module &M) {
  SmallVector<CallInst *, 8> WorkList;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *Call = dyn_cast<CallInst>(&I))
          if (isLowerableLdexp(Call))
            WorkList.push_back(Call);

  for (auto *Call : WorkList) {
    Value *X = Call->getArgOperand(0);
    Value *Exp = Call->getArgOperand(1);
    IRBuilder<> Builder(Call);

    Value *Result;
    if (auto *VecTy = dyn_cast<FixedVectorType>(Call->getType())) {
      // OpenCL does have vector ldexp, but scalarising avoids having to mangle
      // the vector overloads and vector ldexp is vanishingly rare here anyway:
      // it can only come out of the vectorisers, never out of a source builtin.
      Result = PoisonValue::get(VecTy);
      for (unsigned Idx = 0; Idx < VecTy->getNumElements(); ++Idx) {
        Value *LaneX = Builder.CreateExtractElement(X, Idx);
        Value *LaneExp = Exp->getType()->isVectorTy()
                             ? Builder.CreateExtractElement(Exp, Idx)
                             : Exp;
        Value *Lane = emitOpenCLLdexp(Builder, M, LaneX,
                                      coerceLdexpExponent(Builder, LaneExp));
        Result = Builder.CreateInsertElement(Result, Lane, Idx);
      }
    } else {
      Result =
          emitOpenCLLdexp(Builder, M, X, coerceLdexpExponent(Builder, Exp));
    }

    Call->replaceAllUsesWith(Result);
    Call->eraseFromParent();
  }

  return !WorkList.empty();
}

PreservedAnalyses HipLowerRoundIntrinsicsPass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  bool Changed = lowerRoundIntrinsics(M);
  Changed |= lowerLdexpIntrinsics(M);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

#ifndef CHIP_COMBINED_PASS_PLUGIN
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_NAME, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_NAME) {
                    MPM.addPass(HipLowerRoundIntrinsicsPass());
                    return true;
                  }
                  return false;
                });
          }};
}
#endif // CHIP_COMBINED_PASS_PLUGIN
