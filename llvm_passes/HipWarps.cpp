//===- HipWarps.cpp -.-----------------------------------------------------===//
//
// Part of the chipStar Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR pass to handle kernels that are sensitive to warp width.
//
// (c) 2022-2023 Pekka Jääskeläinen / Intel
//===----------------------------------------------------------------------===//
//
// Currently handles kernels that call warp primitives that rely on the
// known warp width by using the reqd_subgroup_size() kernel attribute.
//
// TODO:
// * Lock-step semantics: CUDA/HIP allows dropping explicit thread/WI
// synchronization for cases where warp lock-step semantics guarantees
// a well-defined read-modify-write interleaving inside the warp. We should
// add an annotation that guarantees subgroup lockstep semantics in that case.
// There is not such an OpenCL extension yet to my knowledge.
//===----------------------------------------------------------------------===//

#include "HipWarps.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Metadata.h>
#include "llvm/IR/Module.h"

#include "chipStarConfig.hh"

PreservedAnalyses HipWarpsPass::run(Module &Mod, ModuleAnalysisManager &AM) {

  // We emulate warps with subgroups of which size is implementation and
  // kernel-specific by default while in CUDA/HIP it's a device-specific
  // constant that can be queried from the device info.
  //
  // Add the intel_reqd_sub_group_size kernel metadata to force the subgroup
  // size to be fixed to the warp size used by the chipStar build in case there
  // is a possibility the kernel's semantically sensitive to the warp width.
  //
  // For now check if the CUDA warp-size sensitive intrinsic declarations appear
  // in the module and assume all the kernels call them. TO OPTIMIZE: Use
  // CallGraph to analyze if the kernels really call them to allow subgroup
  // freedom for those that don't.

  std::vector<const char *> WarpSizeSensitiveFuncNames = {
      // Basic shuffle operations
      "_Z6__shfliii",
      "_Z6__shflfii",
      "_Z10__shfl_xoriii",
      "_Z10__shfl_xorfii",
      "_Z9__shfl_upiji",
      "_Z9__shfl_upfji",
      "_Z11__shfl_downiji",
      "_Z11__shfl_downfji",
      
      // Ballot operations
      "_Z8__balloti",
      
      // Subgroup operations
      "_Z16sub_group_balloti",
      "_Z17sub_group_shufflefj",
      "_Z17sub_group_shuffleij",
      "_Z21sub_group_shuffle_xorij",
      "_Z21sub_group_shuffle_xorfj",
      "_Z22sub_group_shuffle_downiij",
      "_Z22sub_group_shuffle_downffj",
      "_Z20sub_group_shuffle_upiij",
      "_Z20sub_group_shuffle_upffj",
      
      // Intel subgroup operations
      "_Z23intel_sub_group_shuffleij",
      "_Z23intel_sub_group_shufflefj",
      "_Z27intel_sub_group_shuffle_xorij",
      "_Z27intel_sub_group_shuffle_xorfj",
      
      // Subgroup ID query
      "_Z22get_sub_group_local_idv",
      
      // Additional warp operations
      "_Z12__chip_allsyncjii",  // __chip_all_sync
      "_Z12__chip_anysyncjii",  // __chip_any_sync
      "_Z15__chip_ballotsynciij", // __chip_ballot_sync
      "_Z9__chip_alli",         // __chip_all
      "_Z9__chip_anyi",         // __chip_any
      "_Z13__chip_balloti",     // __chip_ballot
      "_Z14__chip_lane_idv",    // __chip_lane_id
      "_Z14__chip_syncwarpv",   // __chip_syncwarp
      
      // Sync variants of shuffle operations
      "_Z11__shfl_syncjiii",
      "_Z11__shfl_syncjfii",
      "_Z15__shfl_xor_syncjiiii",
      "_Z15__shfl_xor_syncjfii",
      "_Z14__shfl_up_syncjijii",
      "_Z14__shfl_up_syncjfjii",
      "_Z16__shfl_down_syncjijii",
      "_Z16__shfl_down_syncjfjii"};

  SmallPtrSet<Function *, 8> Sensitive;
  for (auto &FuncName : WarpSizeSensitiveFuncNames)
    if (auto *F = Mod.getFunction(FuncName))
      Sensitive.insert(F);

  if (Sensitive.empty())
    return PreservedAnalyses::all();

  // Kernels that perform an indirect call must not be stamped: the driver then
  // delivers the correct 'this' but zero for every argument after it, silently.
  // No error, no diagnostic, just wrong results.
  //
  // Since one shuffle declaration anywhere in the module stamps every kernel,
  // and Kokkos_Core.hpp declares the shuffle intrinsics, a single #include used
  // to break device-side virtual dispatch for a whole application. See
  // tests/runtime/TestIndirectCall.hip.
  //
  // Everything else keeps the existing conservative behaviour. Narrowing the
  // stamp to kernels that provably reach a sensitive function was tried and is
  // wrong today: WarpSizeSensitiveFuncNames only lists some of the shuffle
  // overloads, so Kokkos' own reductions lost the subgroup size they rely on
  // and started accumulating each contribution several times.
  //
  // A kernel that both dispatches indirectly and shuffles cannot be served
  // either way; correct arguments are the more useful half.
  auto reachesIndirectCall = [](Function &Kernel) {
    SmallPtrSet<Function *, 16> Seen;
    SmallVector<Function *, 16> Worklist{&Kernel};
    Seen.insert(&Kernel);
    while (!Worklist.empty()) {
      Function *F = Worklist.pop_back_val();
      for (Instruction &I : instructions(*F)) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (!CB || CB->isInlineAsm())
          continue;
        Function *Callee = CB->getCalledFunction();
        if (!Callee)
          return true; // Calls through a value: a vtable slot, a callback, ...
        if (!Callee->isDeclaration() && Seen.insert(Callee).second)
          Worklist.push_back(Callee);
      }
    }
    return false;
  };

  auto &Ctx = Mod.getContext();
  for (auto &F : Mod) {
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;
    if (reachesIndirectCall(F))
      continue;

    IntegerType *I32Type = IntegerType::get(Ctx, 32);
    F.setMetadata("intel_reqd_sub_group_size",
                  MDNode::get(Ctx, ConstantAsMetadata::get(ConstantInt::get(
                                       I32Type, CHIP_DEFAULT_WARP_SIZE))));
  }

  // The metadata should not impact other chipStar passes.
  return PreservedAnalyses::all();
}
