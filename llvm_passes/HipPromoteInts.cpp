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
 * Algorithm Overview:
 * ------------------
 * The pass uses a two-phase approach to handle non-standard integer types:
 *
 * 1. Construction Phase:
 *    - When encountering a non-standard integer type (e.g., i33), the pass
 *      first creates a chain of replacement instructions that will eventually 
 *      replace the original ones
 *    - During this phase, intermediate instructions may temporarily use
 *      non-standard types. This is necessary because LLVM requires type 
 *      consistency when building instruction chains
 *    - The pass maintains a map (PromotedValues) tracking both the original and
 *      promoted versions of values to ensure consistent promotion throughout the
 *      chain
 *    - Intermediate zext instructions are created to establish a valid def-use
 *      chain, ensuring instructions get visited and processed later by
 *      promoteChain()
 *
 * 2. Replacement Phase:
 *    - After constructing all necessary instructions, the pass performs the
 *      actual replacements
 *    - All non-standard integer types are promoted to their next larger
 *      standard size (e.g., i33 -> i64)
 *    - The original instructions are replaced with their promoted versions
 *    - The intermediate zext instructions are cleaned up as part of the 
 *      replacement process
 *
 * This two-phase approach is necessary because:
 * 1. LLVM requires type consistency when building instructions
 * 2. We can't modify instructions in place while building their replacements
 * 3. We need to ensure all uses of a value are properly promoted before replacement
 *
 * Initial implementation of this pass used mutateType() which is dangerous and
 * likely to break code.
 *
 * Example kernel that generates non-standard types:
 * __global__ void testWarpCalc(int* debug) {
 *   int tid = threadIdx.x;
 *   int bid = blockIdx.x;
 *   int globalIdx = bid * blockDim.x + tid;
 *
 *   // Optimizations on this loop will generate i33 types.
 *   int result = 0;
 *   for(int i = 0; i < tid + 1; i++) {
 *     result += i * globalIdx;
 *   }
 *
 *   // Store using atomic operation
 *   atomicExch(&debug[globalIdx], result);
 * }
 *
 * https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/2823
 */

using namespace llvm;

bool HipPromoteIntsPass::isStandardBitWidth(unsigned BitWidth) {
  // TODO: 128 is not a standard bit width, will handle later as it's more
  // complex than simply promoting
  return BitWidth == 1 || BitWidth == 8 || BitWidth == 16 || BitWidth == 32 ||
         BitWidth == 64 || BitWidth == 128;
}

unsigned HipPromoteIntsPass::getPromotedBitWidth(unsigned Original) {
  if (Original <= 8)
    return 8;
  if (Original <= 16)
    return 16;
  if (Original <= 32)
    return 32;
  return 64;
}

Type *HipPromoteIntsPass::getPromotedType(Type *TypeToPromote) {
  if (auto *IntTy = dyn_cast<IntegerType>(TypeToPromote)) {
    unsigned PromotedWidth = getPromotedBitWidth(IntTy->getBitWidth());
    return Type::getIntNTy(TypeToPromote->getContext(), PromotedWidth);
  }
  return TypeToPromote; // Return original type if not an integer
}

struct Replacement {
  Instruction *Old;
  Value *New;
  Replacement(Instruction *O, Value *N) : Old(O), New(N) {}
};

void processInstruction(Instruction *I, Type *NonStdType, Type *PromotedTy,
                        const std::string &Indent,
                        SmallVectorImpl<Replacement> &Replacements,
                        SmallDenseMap<Value *, Value *> &PromotedValues) {
  IRBuilder<> Builder(I);

  /// Helper to get or create promoted version of a value
  auto getPromotedValue = [&](Value *V) -> Value * {
    LLVM_DEBUG(dbgs() << Indent << "    getPromotedValue for: " << *V << "\n");

    // First check if we already promoted this value
    if (PromotedValues.count(V)) {
      LLVM_DEBUG(dbgs() << Indent << "      Found existing promotion: "
                        << *PromotedValues[V] << "\n");
      return PromotedValues[V];
    }

    // If it's already the right type, return it
    if (V->getType() == PromotedTy) {
      LLVM_DEBUG(dbgs() << Indent << "      Already correct type: " << *V
                        << "\n");
      return V;
    }

    // If it's the non-standard type, promote it
    if (V->getType() == NonStdType) {
      auto NewV = Builder.CreateZExt(V, PromotedTy);
      PromotedValues[V] = NewV;
      LLVM_DEBUG(dbgs() << Indent << "      Promoting non-standard type: " << *V
                        << " to " << *NewV << "\n");
      return NewV;
    }

    // Otherwise return original value
    LLVM_DEBUG(dbgs() << Indent << "      Using original value: " << *V
                      << "\n");
    return V;
  };

  if (isa<PHINode>(I)) {
    PHINode *Phi = cast<PHINode>(I);
    // Create new PHI node with the promoted type (e.g., i64) instead of
    // original type
    Type *PromotedType = HipPromoteIntsPass::getPromotedType(Phi->getType());
    PHINode *NewPhi =
        PHINode::Create(PromotedType, Phi->getNumIncomingValues(), "", Phi);

    // Copy all incoming values and blocks
    for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
      Value *IncomingValue = Phi->getIncomingValue(i);
      BasicBlock *IncomingBlock = Phi->getIncomingBlock(i);

      // If the incoming value is from our promotion chain, use the promoted
      // value
      Value *NewIncomingValue = PromotedValues.count(IncomingValue)
                                    ? PromotedValues[IncomingValue]
                                    : IncomingValue;

      // If the incoming value isn't promoted yet, promote it now
      if (NewIncomingValue->getType() != PromotedType)
        NewIncomingValue = Builder.CreateZExt(NewIncomingValue, PromotedType);

      NewPhi->addIncoming(NewIncomingValue, IncomingBlock);
    }

    LLVM_DEBUG(dbgs() << Indent << "  " << *I << "    ============> " << *NewPhi
                      << "\n");
    PromotedValues[Phi] = NewPhi;
    Replacements.push_back(Replacement(I, NewPhi));
  } else if (isa<ZExtInst>(I)) {
    ZExtInst *ZExtI = cast<ZExtInst>(I);
    Value *SrcOp = ZExtI->getOperand(0);

    // If we're extending from our non-standard type to our promoted type,
    // just use the promoted value directly
    if (SrcOp->getType() == NonStdType && ZExtI->getDestTy() == PromotedTy) {
      Value *PromotedSrc =
          PromotedValues.count(SrcOp) ? PromotedValues[SrcOp] : SrcOp;
      LLVM_DEBUG(dbgs() << Indent << "  " << *I
                        << "    ============> Using promoted: " << *PromotedSrc
                        << "\n");
      PromotedValues[I] = PromotedSrc;
      Replacements.push_back(Replacement(I, PromotedSrc));
    } else {
      // Otherwise handle as normal
      Value *PromotedSrc =
          PromotedValues.count(SrcOp) ? PromotedValues[SrcOp] : SrcOp;
      if (PromotedSrc->getType() != PromotedTy) {
        PromotedSrc = Builder.CreateZExt(PromotedSrc, PromotedTy);
      }
      PromotedValues[I] = PromotedSrc;
      Replacements.push_back(Replacement(I, PromotedSrc));
      LLVM_DEBUG(dbgs() << Indent << "  " << *I << "    ============> "
                        << *PromotedSrc << "\n");
    }
  } else if (isa<TruncInst>(I)) {
    TruncInst *TruncI = cast<TruncInst>(I);
    Value *SrcOp = TruncI->getOperand(0);
    Value *PromotedSrc =
        PromotedValues.count(SrcOp) ? PromotedValues[SrcOp] : SrcOp;

    // Verify the source is actually of our promoted type
    if (PromotedSrc->getType() != PromotedTy)
      PromotedSrc = Builder.CreateZExt(PromotedSrc, PromotedTy);

    // Create a new trunc for external users
    Value *NewTrunc = Builder.CreateTrunc(PromotedSrc, TruncI->getType());
    LLVM_DEBUG(dbgs() << Indent << "  " << *I << "    ============> "
                      << *NewTrunc << "\n");

    // Store both the promoted and truncated versions
    PromotedValues[I] = PromotedSrc; // Use promoted version in our chain
    Replacements.push_back(Replacement(
        I,
        NewTrunc)); // Replace old instruction with new trunc for external users
  } else if (isa<BinaryOperator>(I)) {
    BinaryOperator *BinOp = cast<BinaryOperator>(I);
    bool NeedsPromotion = (BinOp->getType() == NonStdType);

    Value *LHS = getPromotedValue(BinOp->getOperand(0));
    Value *RHS = getPromotedValue(BinOp->getOperand(1));

    Value *NewInst;
    if (NeedsPromotion) {
      // Create operation in promoted type
      NewInst = Builder.CreateBinOp(BinOp->getOpcode(), LHS, RHS);
    } else {
      // For operations that should stay in original type
      if (LHS->getType() != BinOp->getType())
        LHS = Builder.CreateTrunc(LHS, BinOp->getType());
      if (RHS->getType() != BinOp->getType())
        RHS = Builder.CreateTrunc(RHS, BinOp->getType());
      NewInst = Builder.CreateBinOp(BinOp->getOpcode(), LHS, RHS);
    }

    LLVM_DEBUG(dbgs() << Indent << "  " << *I << "    ============> "
                      << *NewInst << "\n");
    PromotedValues[I] = NewInst;
    Replacements.push_back(Replacement(I, NewInst));
  } else if (isa<CallInst>(I)) {
    CallInst *OldCall = cast<CallInst>(I);
    // Create a new call with the same operands, but use promoted values where
    // available
    SmallVector<Value *, 8> NewArgs;
    for (unsigned i = 0; i < OldCall->arg_size(); ++i) {
      Value *OldArg = OldCall->getArgOperand(i);
      Value *NewArg =
          PromotedValues.count(OldArg) ? PromotedValues[OldArg] : OldArg;

      // if the function expects a non-standard type, abort for now.
      // TODO: if this assert is hit, we need to handle this case in the future
      // by promoting the function arguments as well.
      if (OldArg->getType() != NewArg->getType())
        assert(false &&
               "HipPromoteIntsPass: Function expects non-standard type");

      NewArgs.push_back(NewArg);
    }

    CallInst *NewCall = CallInst::Create(OldCall->getFunctionType(),
                                         OldCall->getCalledOperand(), NewArgs,
                                         OldCall->getName(), OldCall);
    NewCall->setCallingConv(OldCall->getCallingConv());
    NewCall->setAttributes(OldCall->getAttributes());

    LLVM_DEBUG(dbgs() << Indent << "  " << *I << "    ============> "
                      << *NewCall << "\n");
    PromotedValues[I] = NewCall;
    Replacements.push_back(Replacement(I, NewCall));
  } else if (isa<ReturnInst>(I)) {
    ReturnInst *RetI = cast<ReturnInst>(I);
    
    // If there's a return value, check if it needs to be promoted
    if (RetI->getNumOperands() > 0) {
      Value *RetVal = RetI->getOperand(0);
      Value *PromotedRetVal = PromotedValues.count(RetVal) ? PromotedValues[RetVal] : RetVal;
      
      // Make sure the return value matches the function's return type
      if (PromotedRetVal->getType() != I->getFunction()->getReturnType()) {
        // If the function return type is larger than the value type, extend it
        if (PromotedRetVal->getType()->getPrimitiveSizeInBits() < 
            I->getFunction()->getReturnType()->getPrimitiveSizeInBits()) {
          PromotedRetVal = Builder.CreateZExt(PromotedRetVal, I->getFunction()->getReturnType());
        } 
        // If the function return type is smaller, truncate it
        else if (PromotedRetVal->getType()->getPrimitiveSizeInBits() > 
                 I->getFunction()->getReturnType()->getPrimitiveSizeInBits()) {
          PromotedRetVal = Builder.CreateTrunc(PromotedRetVal, I->getFunction()->getReturnType());
        }
      }
      
      // Create a new return instruction with the correctly typed value
      ReturnInst *NewRet = Builder.CreateRet(PromotedRetVal);
      
      LLVM_DEBUG(dbgs() << Indent << "  " << *I << "    ============> "
                        << *NewRet << "\n");
      PromotedValues[I] = NewRet;
      Replacements.push_back(Replacement(I, NewRet));
    } else {
      // Handle void return
      ReturnInst *NewRet = Builder.CreateRetVoid();
      LLVM_DEBUG(dbgs() << Indent << "  " << *I << "    ============> "
                        << *NewRet << "\n");
      PromotedValues[I] = NewRet;
      Replacements.push_back(Replacement(I, NewRet));
    }
  } else {
    LLVM_DEBUG(dbgs() << Indent << "  Unhandled instruction type: " << *I << "\n");
    assert(false && "HipPromoteIntsPass: Unhandled instruction type");
  }
}

static void promoteChain(Instruction *OldI, Type *NonStdType, Type *PromotedTy,
                         SmallPtrSetImpl<Instruction *> &Visited,
                         SmallVectorImpl<Replacement> &Replacements,
                         SmallDenseMap<Value *, Value *> &PromotedValues,
                         unsigned Depth = 0) {
  // If we've already processed this instruction, just return
  if (!Visited.insert(OldI).second) {
    // If we have a promoted value for this instruction, use it
    if (PromotedValues.count(OldI))
      LLVM_DEBUG(dbgs() << std::string(Depth * 2, ' ')
                        << "Already processed: " << *OldI << "\n");
    return;
  }

  std::string Indent(Depth * 2, ' ');

  // Process instruction
  processInstruction(OldI, NonStdType, PromotedTy, Indent, Replacements,
                     PromotedValues);

  // Recursively process all users
  for (User *U : OldI->users())
    if (auto *UI = dyn_cast<Instruction>(U))
      promoteChain(UI, NonStdType, PromotedTy, Visited, Replacements,
                   PromotedValues, Depth + 1);

  return;
}

PreservedAnalyses HipPromoteIntsPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  bool Changed = false;
  SmallPtrSet<Instruction *, 32>
      GlobalVisited; // Track all visited instructions across chains

  for (Function &F : M) {
    SmallVector<Instruction *, 16> WorkList;

    // First collect all instructions we need to promote
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *IntTy = dyn_cast<IntegerType>(I.getType()))
          if (!isStandardBitWidth(IntTy->getBitWidth()))
            WorkList.push_back(&I);

    // Process the worklist
    for (Instruction *I : WorkList) {
      // Skip if we've already processed this instruction as part of another
      // chain
      if (GlobalVisited.count(I))
        continue;

      if (auto *IntTy = dyn_cast<IntegerType>(I->getType())) {
        if (!isStandardBitWidth(IntTy->getBitWidth())) {
          unsigned PromotedBitWidth = getPromotedBitWidth(IntTy->getBitWidth());
          Type *PromotedType =
              Type::getIntNTy(M.getContext(), PromotedBitWidth);

          SmallVector<Replacement, 16> Replacements;
          SmallDenseMap<Value *, Value *> PromotedValues;

          // Use GlobalVisited instead of creating a new set
          promoteChain(I, IntTy, PromotedType, GlobalVisited, Replacements,
                       PromotedValues, 0);

          // Update uses and cleanup as before
          for (const auto &R : Replacements) {
            for (auto &U : R.Old->uses()) {
              User *User = U.getUser();
              if (!GlobalVisited.count(cast<Instruction>(User)))
                U.set(R.New);
            }
          }

          for (auto It = Replacements.rbegin(); It != Replacements.rend();
               ++It)
            It->Old->eraseFromParent();

          Changed = true;
        }
      }
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}