#include "HipPromoteInts.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "hip-promote-ints"

/**
 * This pass promotes integer types to the next standard bit width.
 * During optimization of loops, LLVM generates non-standard integer types
 * such as i33 or i56
 * 
 * __global__ void testWarpCalc(int* debug) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int globalIdx = bid * blockDim.x + tid;
    
    // Optimizations on this loop will generate i33 types.
    int result = 0;
    for(int i = 0; i < tid + 1; i++) {
        result += i * globalIdx;
    }
    
    // Store using atomic operation
    atomicExch(&debug[globalIdx], result);
}
 * 
 * https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/2823
 */

using namespace llvm;

bool HipPromoteIntsPass::isStandardBitWidth(unsigned BitWidth) {
  return BitWidth == 1 || BitWidth == 8 || BitWidth == 16 || BitWidth == 32 || BitWidth == 64;
}

unsigned HipPromoteIntsPass::getPromotedBitWidth(unsigned Original) {
  if (Original <= 8) return 8;
  if (Original <= 16) return 16;
  if (Original <= 32) return 32;
  return 64;
}

PreservedAnalyses HipPromoteIntsPass::run(Module &M, ModuleAnalysisManager &AM) {
  bool Changed = false;
  
  for (Function &F : M) {
    LLVM_DEBUG(dbgs() << "[HipPromoteInts] Analyzing function: " << F.getName() << "\n");
    
    for (BasicBlock &BB : F) {
      // Use a vector to store instructions that need modification
      std::vector<Instruction*> WorkList;
      for (Instruction &I : BB) {
        WorkList.push_back(&I);
      }
      
      // Process the worklist safely outside the BB iteration
      for (Instruction *I : WorkList) {
        if (auto *IntTy = dyn_cast<IntegerType>(I->getType())) {
          if (!isStandardBitWidth(IntTy->getBitWidth())) {
            LLVM_DEBUG(dbgs() << "[HipPromoteInts] Found non-standard type in result: " << *I << "\n");
            
            unsigned NextStdSize = getPromotedBitWidth(IntTy->getBitWidth());
            Type *PromotedType = Type::getIntNTy(M.getContext(), NextStdSize);
            
            LLVM_DEBUG(dbgs() << "[HipPromoteInts] Promoting from i" << IntTy->getBitWidth() 
                      << " to i" << NextStdSize << "\n");
            
            // Update the instruction type
            I->mutateType(PromotedType);
            
            // Special handling for trunc instructions where source and dest are same size
            if (isa<TruncInst>(I)) {
              auto *Trunc = cast<TruncInst>(I);
              Value *Src = Trunc->getOperand(0);
              if (auto *SrcIntTy = dyn_cast<IntegerType>(Src->getType())) {
                if (SrcIntTy->getBitWidth() == NextStdSize) {
                  LLVM_DEBUG(dbgs() << "[HipPromoteInts] Found trunc with matching source size: " << *Trunc << "\n");
                  LLVM_DEBUG(dbgs() << "[HipPromoteInts] Source operand: " << *Src << "\n");
                  // When source and dest types are the same, just use the source directly
                  Trunc->replaceAllUsesWith(Src);
                  Trunc->eraseFromParent();
                  Changed = true;
                  continue;
                }
              }
            }
            
            // Update operands if needed
            if (auto *BinOp = dyn_cast<BinaryOperator>(I)) {
              Value *LHS = BinOp->getOperand(0);
              Value *RHS = BinOp->getOperand(1);
              
              IRBuilder<> Builder(I);
              if (LHS->getType() != PromotedType) {
                LHS = Builder.CreateZExtOrTrunc(LHS, PromotedType);
                BinOp->setOperand(0, LHS);
              }
              if (RHS->getType() != PromotedType) {
                RHS = Builder.CreateZExtOrTrunc(RHS, PromotedType);
                BinOp->setOperand(1, RHS);
              }
            }
            
            LLVM_DEBUG(dbgs() << "[HipPromoteInts] Instruction after promotion: " << *I << "\n");
            Changed = true;
          }
        }
      }
    }
  }
  
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
} 