//===- HipAbort.cpp -------------------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// LLVM IR pass to handle kernels with abort() calls.
//
// (c) 2023 CHIP-SPV developers
//     2022 Pekka Jääskeläinen / Parmance for Argonne National Laboratory
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_HIP_ABORT_H
#define LLVM_PASSES_HIP_ABORT_H

#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"

#include <map>

using namespace llvm;

class HipAbortPass : public PassInfoMixin<HipAbortPass> {
public:
  // Rules:
  //   Unresolved + WontAbort -> Unresolved.
  //    MayAbort + <anything> -> MayAbort.
  enum AbortAttribute { Unresolved = 0, WontAbort, MayAbort };

  struct InverseCallGraphNode {
    const CallGraphNode *OrigNode = nullptr;
    std::set<InverseCallGraphNode *> Callers;
    AbortAttribute AbortAttr = Unresolved;
    InverseCallGraphNode(const CallGraphNode *TheOrigNode)
        : OrigNode(TheOrigNode) {}
    Function *getFunction() const { return OrigNode->getFunction(); }
    bool mayCallAbort() const { return AbortAttr == AbortAttribute::MayAbort; }
  };

private:
  std::map<Function *, InverseCallGraphNode *> InverseCallGraph;

  /// True if any kernel may call abort().
  bool AnyKernelMayCallAbort = false;

public:
  ~HipAbortPass() { reset(); }

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
  void reset() {
    for (auto KV : InverseCallGraph)
      delete KV.second;
    InverseCallGraph.clear();
    AnyKernelMayCallAbort = false;
  }

private:
  InverseCallGraphNode *getInvertedCGNode(Function *F);
  InverseCallGraphNode *getInvertedCGNode(const CallGraphNode *CGN);
  InverseCallGraphNode *getOrCreateInvertedCGNode(const CallGraphNode *CGN);
  bool mayCallAbort(const CallInst *CI) const;
  void buildInvertedCallGraph(const CallGraph &CG);
  void analyze(const CallGraph &CG);
  void processFunctions(Module &M);
};

#endif
