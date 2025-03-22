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

  /// Helper to get or create promoted value
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
      // Make sure we don't try to extend a larger type to a smaller one
      Value *NewV = nullptr;
      if (V->getType()->getPrimitiveSizeInBits() < PromotedTy->getPrimitiveSizeInBits()) {
        NewV = Builder.CreateZExt(V, PromotedTy);
        LLVM_DEBUG(dbgs() << Indent << "      Promoting non-standard type with zext: " << *V
                          << " to " << *NewV << "\n");
      } else if (V->getType()->getPrimitiveSizeInBits() > PromotedTy->getPrimitiveSizeInBits()) {
        NewV = Builder.CreateTrunc(V, PromotedTy);
        LLVM_DEBUG(dbgs() << Indent << "      Promoting non-standard type with trunc: " << *V
                          << " to " << *NewV << "\n");
      } else {
        NewV = Builder.CreateBitCast(V, PromotedTy);
        LLVM_DEBUG(dbgs() << Indent << "      Promoting non-standard type with bitcast: " << *V
                          << " to " << *NewV << "\n");
      }
      PromotedValues[V] = NewV;
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
        
    // Register the PHI node in the map BEFORE processing incoming values
    // to handle circular references properly
    LLVM_DEBUG(dbgs() << Indent << "  Creating promotion for PHI: " << *Phi
                      << " to " << *NewPhi << "\n");
    PromotedValues[Phi] = NewPhi;

    // Copy all incoming values and blocks
    for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
      Value *IncomingValue = Phi->getIncomingValue(i);
      BasicBlock *IncomingBlock = Phi->getIncomingBlock(i);

      LLVM_DEBUG(dbgs() << Indent << "    Processing incoming value: " << *IncomingValue
                        << " from block: " << IncomingBlock->getName() << "\n");
                        
      Value *NewIncomingValue = nullptr;
      
      // Special handling for truncation instructions with non-standard intermediate types
      if (auto *TruncI = dyn_cast<TruncInst>(IncomingValue)) {
        Value *TruncSrc = TruncI->getOperand(0);
        
        // Check if the source operand has a non-standard type
        if (auto *SrcTy = dyn_cast<IntegerType>(TruncSrc->getType())) {
          if (!HipPromoteIntsPass::isStandardBitWidth(SrcTy->getBitWidth())) {
            LLVM_DEBUG(dbgs() << Indent << "      Found truncation from non-standard type: " << *TruncI << "\n");
            
            // Instead of creating a new truncation chain, we need to handle the chain consistently
            // First, get the mapped value for this truncation instruction if it exists
            if (PromotedValues.count(IncomingValue)) {
              NewIncomingValue = PromotedValues[IncomingValue];
              LLVM_DEBUG(dbgs() << Indent << "      Using existing promoted value for truncation: " 
                               << *NewIncomingValue << "\n");
            } else {
              // If this truncation hasn't been processed yet, we should let the normal truncation
              // handling take care of it later in promoteChain
              // For now, just use the original value to avoid inconsistencies
              NewIncomingValue = IncomingValue;
              LLVM_DEBUG(dbgs() << Indent << "      Using original truncation value to maintain consistency\n");
            }
          }
        }
      }
      // Also handle zero-extension from non-standard types
      else if (auto *ZExtI = dyn_cast<ZExtInst>(IncomingValue)) {
        Value *ZExtSrc = ZExtI->getOperand(0);
        
        // Check if the source operand has a non-standard type
        if (auto *SrcTy = dyn_cast<IntegerType>(ZExtSrc->getType())) {
          if (!HipPromoteIntsPass::isStandardBitWidth(SrcTy->getBitWidth())) {
            LLVM_DEBUG(dbgs() << Indent << "      Found zero-extension from non-standard type: " << *ZExtI << "\n");
            
            // Handle consistently with how we process ZExt instructions
            if (PromotedValues.count(IncomingValue)) {
              NewIncomingValue = PromotedValues[IncomingValue];
              LLVM_DEBUG(dbgs() << Indent << "      Using existing promoted value for zext: " 
                               << *NewIncomingValue << "\n");
            } else {
              // If this zext hasn't been processed yet, let the normal processing handle it
              NewIncomingValue = IncomingValue;
              LLVM_DEBUG(dbgs() << Indent << "      Using original zero-extension value to maintain consistency\n");
            }
          }
        }
      }
      
      // If not handled by special case above, use normal promotion
      if (!NewIncomingValue) {
        // If the incoming value is from our promotion chain, use the promoted value
        NewIncomingValue = PromotedValues.count(IncomingValue)
                                ? PromotedValues[IncomingValue]
                                : IncomingValue;

        // If the incoming value isn't promoted yet, promote it now
        if (NewIncomingValue->getType() != PromotedType) {
          // Check if we need to extend or truncate based on bit size
          if (NewIncomingValue->getType()->getPrimitiveSizeInBits() < PromotedType->getPrimitiveSizeInBits()) {
            LLVM_DEBUG(dbgs() << Indent << "      zExting incoming value: " << *IncomingValue
                              << " ===> " << *NewIncomingValue << "\n");
            NewIncomingValue = Builder.CreateZExt(NewIncomingValue, PromotedType);
          } else if (NewIncomingValue->getType()->getPrimitiveSizeInBits() > PromotedType->getPrimitiveSizeInBits()) {
            LLVM_DEBUG(dbgs() << Indent << "      truncing incoming value: " << *IncomingValue
                              << " ===> " << *NewIncomingValue << "\n");
            NewIncomingValue = Builder.CreateTrunc(NewIncomingValue, PromotedType);
          } else {
            // Same bit width but different types
            LLVM_DEBUG(dbgs() << Indent << "      bitcasting incoming value: " << *IncomingValue
                              << " ===> " << *NewIncomingValue << "\n");
            NewIncomingValue = Builder.CreateBitCast(NewIncomingValue, PromotedType);
          }
        }
      }

      NewPhi->addIncoming(NewIncomingValue, IncomingBlock);
    }

    LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting PHI node: ====> " << *NewPhi
                      << "\n");
    Replacements.push_back(Replacement(Phi, NewPhi));
  } else if (isa<ZExtInst>(I)) {
    ZExtInst *ZExtI = cast<ZExtInst>(I);
    Value *SrcOp = ZExtI->getOperand(0);

    // Get promoted source if available
    Value *PromotedSrc =
        PromotedValues.count(SrcOp) ? PromotedValues[SrcOp] : SrcOp;

    // Handle zero extension based on source and destination types
    if (SrcOp->getType() == NonStdType) {
      // If the source is our non-standard type, we need to be careful with type consistency
      LLVM_DEBUG(dbgs() << Indent << "  ZExt with non-standard source type: " << *ZExtI << "\n");
      
      // Create a new zext with properly promoted types
      Type *DestTy = ZExtI->getDestTy();
      
      // If destination type is smaller than our promoted type, we need to truncate first
      if (DestTy->getPrimitiveSizeInBits() < PromotedTy->getPrimitiveSizeInBits()) {
        Value *NewZExt = Builder.CreateTrunc(PromotedSrc, DestTy);
        LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting ZExt (with trunc): ====> "
                          << *NewZExt << "\n");
        PromotedValues[I] = NewZExt;
        Replacements.push_back(Replacement(I, NewZExt));
      } 
      // If destination type is larger, create a proper zext
      else if (DestTy->getPrimitiveSizeInBits() > PromotedTy->getPrimitiveSizeInBits()) {
        Value *NewZExt = Builder.CreateZExt(PromotedSrc, DestTy);
        LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting ZExt (with zext): ====> "
                          << *NewZExt << "\n");
        PromotedValues[I] = NewZExt;
        Replacements.push_back(Replacement(I, NewZExt));
      }
      // If destination is exactly our promoted type, just use the promoted source
      else {
        LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting ZExt (direct): ====> "
                          << *PromotedSrc << "\n");
        PromotedValues[I] = PromotedSrc;
        Replacements.push_back(Replacement(I, PromotedSrc));
      }
    } else {
      // For standard source types, handle normally
      if (PromotedSrc->getType() != PromotedTy && 
          PromotedSrc->getType()->getPrimitiveSizeInBits() < PromotedTy->getPrimitiveSizeInBits()) {
        PromotedSrc = Builder.CreateZExt(PromotedSrc, PromotedTy);
      }
      
      // Create a new zext to the destination type
      Type *DestTy = ZExtI->getDestTy();
      Value *NewZExt;
      
      if (PromotedSrc->getType()->getPrimitiveSizeInBits() > DestTy->getPrimitiveSizeInBits()) {
        NewZExt = Builder.CreateTrunc(PromotedSrc, DestTy);
      } else if (PromotedSrc->getType()->getPrimitiveSizeInBits() < DestTy->getPrimitiveSizeInBits()) {
        NewZExt = Builder.CreateZExt(PromotedSrc, DestTy);
      } else {
        NewZExt = PromotedSrc; // Same size, no conversion needed
      }
      
      LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting ZExt: ====> "
                       << *NewZExt << "\n");
      PromotedValues[I] = NewZExt;
      Replacements.push_back(Replacement(I, NewZExt));
    }
  } else if (isa<TruncInst>(I)) {
    TruncInst *TruncI = cast<TruncInst>(I);
    Value *SrcOp = TruncI->getOperand(0);
    Value *PromotedSrc =
        PromotedValues.count(SrcOp) ? PromotedValues[SrcOp] : SrcOp;

    // Verify the source is actually of our promoted type
    if (PromotedSrc->getType() != PromotedTy) {
      // Check if we need to extend or truncate
      if (PromotedSrc->getType()->getPrimitiveSizeInBits() < PromotedTy->getPrimitiveSizeInBits()) {
        LLVM_DEBUG(dbgs() << Indent << "    ZExting source: " << *PromotedSrc << " to " << *PromotedTy << "\n");
        PromotedSrc = Builder.CreateZExt(PromotedSrc, PromotedTy);
      } else if (PromotedSrc->getType()->getPrimitiveSizeInBits() > PromotedTy->getPrimitiveSizeInBits()) {
        LLVM_DEBUG(dbgs() << Indent << "    Truncing source: " << *PromotedSrc << " to " << *PromotedTy << "\n");
        PromotedSrc = Builder.CreateTrunc(PromotedSrc, PromotedTy);
      } else {
        // Same bit width but different types, this should rarely happen
        LLVM_DEBUG(dbgs() << Indent << "    Bitcasting source: " << *PromotedSrc << " to " << *PromotedTy << "\n");
        PromotedSrc = Builder.CreateBitCast(PromotedSrc, PromotedTy);
      }
    }

    // Check if we're truncating to a non-standard type
    Type *DestTy = TruncI->getDestTy();
    if (auto *IntTy = dyn_cast<IntegerType>(DestTy)) {
      if (!HipPromoteIntsPass::isStandardBitWidth(IntTy->getBitWidth())) {
        // Instead of truncating to a non-standard type, truncate to the next LARGER standard type
        unsigned NewWidth = HipPromoteIntsPass::getPromotedBitWidth(IntTy->getBitWidth());
        DestTy = Type::getIntNTy(DestTy->getContext(), NewWidth);
        LLVM_DEBUG(dbgs() << Indent << "    Changing truncation target from i" 
                  << IntTy->getBitWidth() << " to i" << NewWidth << "\n");
      }
    }

    // Create a new trunc for external users
    Value *NewTrunc = Builder.CreateTrunc(PromotedSrc, DestTy);
    LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting Trunc: ====> "
                      << *NewTrunc << "\n");

    // Store both the promoted and truncated versions
    PromotedValues[I] = NewTrunc; // Use truncated version as the default
    
    // Only use the promoted version for internal operations on non-standard types
    if (auto *IntTy = dyn_cast<IntegerType>(DestTy)) {
      if (!HipPromoteIntsPass::isStandardBitWidth(IntTy->getBitWidth())) {
        // For non-standard destination types, store the promoted version for our internal use
        PromotedValues[I] = PromotedSrc;
      }
    }
    
    Replacements.push_back(Replacement(I, NewTrunc));
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

    LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting BinOp: ====> "
                      << *NewInst << "\n");
    PromotedValues[I] = NewInst;
    Replacements.push_back(Replacement(I, NewInst));
  } else if (isa<SelectInst>(I)) {
    SelectInst *SelI = cast<SelectInst>(I);
    bool NeedsPromotion = (SelI->getType() == NonStdType);
    
    // Get promoted operands
    Value *Condition = getPromotedValue(SelI->getCondition());
    Value *TrueVal = getPromotedValue(SelI->getTrueValue());
    Value *FalseVal = getPromotedValue(SelI->getFalseValue());
    
    // Make sure condition is i1
    if (Condition->getType() != Type::getInt1Ty(I->getContext())) {
      LLVM_DEBUG(dbgs() << Indent << "    Converting condition to i1: " << *Condition << "\n");
      Condition = Builder.CreateICmpNE(
          Condition, 
          Constant::getNullValue(Condition->getType()),
          "select.cond");
    }
    
    Value *NewSelect;
    if (NeedsPromotion) {
      // Create operation in promoted type
      NewSelect = Builder.CreateSelect(Condition, TrueVal, FalseVal, SelI->getName());
    } else {
      // For operations that should stay in original type
      Type *OriginalType = SelI->getType();
      
      // True and false values must match the select's type
      if (TrueVal->getType() != OriginalType) {
        if (TrueVal->getType()->getPrimitiveSizeInBits() > OriginalType->getPrimitiveSizeInBits()) {
          TrueVal = Builder.CreateTrunc(TrueVal, OriginalType);
          LLVM_DEBUG(dbgs() << Indent << "    Truncating true value to match select type\n");
        } else if (TrueVal->getType()->getPrimitiveSizeInBits() < OriginalType->getPrimitiveSizeInBits()) {
          TrueVal = Builder.CreateZExt(TrueVal, OriginalType);
          LLVM_DEBUG(dbgs() << Indent << "    Extending true value to match select type\n");
        } else {
          TrueVal = Builder.CreateBitCast(TrueVal, OriginalType);
          LLVM_DEBUG(dbgs() << Indent << "    Bitcasting true value to match select type\n");
        }
      }
      
      if (FalseVal->getType() != OriginalType) {
        if (FalseVal->getType()->getPrimitiveSizeInBits() > OriginalType->getPrimitiveSizeInBits()) {
          FalseVal = Builder.CreateTrunc(FalseVal, OriginalType);
          LLVM_DEBUG(dbgs() << Indent << "    Truncating false value to match select type\n");
        } else if (FalseVal->getType()->getPrimitiveSizeInBits() < OriginalType->getPrimitiveSizeInBits()) {
          FalseVal = Builder.CreateZExt(FalseVal, OriginalType);
          LLVM_DEBUG(dbgs() << Indent << "    Extending false value to match select type\n");
        } else {
          FalseVal = Builder.CreateBitCast(FalseVal, OriginalType);
          LLVM_DEBUG(dbgs() << Indent << "    Bitcasting false value to match select type\n");
        }
      }
      
      NewSelect = Builder.CreateSelect(Condition, TrueVal, FalseVal, SelI->getName());
    }
    
    LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting Select: ====> "
                      << *NewSelect << "\n");
    PromotedValues[I] = NewSelect;
    Replacements.push_back(Replacement(I, NewSelect));
  } else if (isa<ICmpInst>(I)) {
    ICmpInst *CmpI = cast<ICmpInst>(I);
    
    // Get promoted operands
    Value *LHS = getPromotedValue(CmpI->getOperand(0));
    Value *RHS = getPromotedValue(CmpI->getOperand(1));
    
    // Make sure operands are of same type for comparison
    if (LHS->getType() != RHS->getType()) {
      // If one operand is promoted and the other isn't, promote the other
      if (LHS->getType() == PromotedTy) {
        RHS = Builder.CreateZExt(RHS, PromotedTy);
      } else if (RHS->getType() == PromotedTy) {
        LHS = Builder.CreateZExt(LHS, PromotedTy);
      } else {
        // If neither is promoted type but they still differ, convert to common type
        Type *CommonType = LHS->getType()->getPrimitiveSizeInBits() > 
                           RHS->getType()->getPrimitiveSizeInBits() ? 
                           LHS->getType() : RHS->getType();
        if (LHS->getType() != CommonType)
          LHS = Builder.CreateZExt(LHS, CommonType);
        if (RHS->getType() != CommonType)
          RHS = Builder.CreateZExt(RHS, CommonType);
      }
    }
    
    // Create new comparison instruction
    Value *NewCmp = Builder.CreateICmp(CmpI->getPredicate(), LHS, RHS, CmpI->getName());
    
    LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting ICmp: ====> "
                      << *NewCmp << "\n");
    PromotedValues[I] = NewCmp;
    Replacements.push_back(Replacement(I, NewCmp));
  } else if (isa<CallInst>(I)) {
    CallInst *OldCall = cast<CallInst>(I);
    // Create a new call with the same operands, but use promoted values where
    // available
    SmallVector<Value *, 8> NewArgs;
    for (unsigned i = 0; i < OldCall->arg_size(); ++i) {
      Value *OldArg = OldCall->getArgOperand(i);
      Value *NewArg =
          PromotedValues.count(OldArg) ? PromotedValues[OldArg] : OldArg;

      // If the argument type doesn't match what the function expects,
      // we need to convert it back to the expected type
      Type *ExpectedType = OldCall->getFunctionType()->getParamType(i);
      if (NewArg->getType() != ExpectedType) {
        // Check if we need to truncate or extend
        if (auto *IntTy = dyn_cast<IntegerType>(ExpectedType)) {
          if (auto *ArgIntTy = dyn_cast<IntegerType>(NewArg->getType())) {
            // If expected type is smaller, truncate
            if (IntTy->getBitWidth() < ArgIntTy->getBitWidth()) {
              NewArg = Builder.CreateTrunc(NewArg, ExpectedType);
              LLVM_DEBUG(dbgs() << Indent << "    Truncating argument from " 
                       << *ArgIntTy << " to " << *IntTy << "\n");
            }
            // If expected type is larger, extend
            else if (IntTy->getBitWidth() > ArgIntTy->getBitWidth()) {
              NewArg = Builder.CreateZExt(NewArg, ExpectedType);
              LLVM_DEBUG(dbgs() << Indent << "    Extending argument from " 
                       << *ArgIntTy << " to " << *IntTy << "\n");
            }
          }
        }
      }

      NewArgs.push_back(NewArg);
    }

    CallInst *NewCall = CallInst::Create(OldCall->getFunctionType(),
                                         OldCall->getCalledOperand(), NewArgs,
                                         OldCall->getName(), OldCall);
    NewCall->setCallingConv(OldCall->getCallingConv());
    NewCall->setAttributes(OldCall->getAttributes());

    LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting Call: ====> "
                      << *NewCall << "\n");
    PromotedValues[I] = NewCall;
    Replacements.push_back(Replacement(I, NewCall));
  } else if (isa<StoreInst>(I)) {
    StoreInst *Store = cast<StoreInst>(I);
    
    // Get the value being stored (possibly promoted)
    Value *StoredValue = Store->getValueOperand();
    Value *NewStoredValue = PromotedValues.count(StoredValue) 
                          ? PromotedValues[StoredValue] 
                          : StoredValue;
    
    // Get the pointer (we don't normally promote pointers)
    Value *Ptr = Store->getPointerOperand();
    
    // Check if the value type needs adjustment to match what's expected by the store
    Type *ExpectedType = StoredValue->getType();
    if (NewStoredValue->getType() != ExpectedType) {
      // If the promoted value is larger, truncate it back
      if (NewStoredValue->getType()->getPrimitiveSizeInBits() > ExpectedType->getPrimitiveSizeInBits()) {
        LLVM_DEBUG(dbgs() << Indent << "    Truncating store value from " 
                         << *NewStoredValue->getType() << " to " << *ExpectedType << "\n");
        NewStoredValue = Builder.CreateTrunc(NewStoredValue, ExpectedType);
      }
      // If it's smaller (unusual), extend it
      else if (NewStoredValue->getType()->getPrimitiveSizeInBits() < ExpectedType->getPrimitiveSizeInBits()) {
        LLVM_DEBUG(dbgs() << Indent << "    Extending store value from " 
                         << *NewStoredValue->getType() << " to " << *ExpectedType << "\n");
        NewStoredValue = Builder.CreateZExt(NewStoredValue, ExpectedType);
      }
      // If same size but different types, bitcast
      else {
        LLVM_DEBUG(dbgs() << Indent << "    Bitcasting store value to match original type\n");
        NewStoredValue = Builder.CreateBitCast(NewStoredValue, ExpectedType);
      }
    }
    
    // Create a new store instruction
    StoreInst *NewStore = Builder.CreateStore(NewStoredValue, Ptr);
    
    // Preserve the alignment and other attributes from the original store
    NewStore->setAlignment(Store->getAlign());
    NewStore->setVolatile(Store->isVolatile());
    NewStore->setOrdering(Store->getOrdering());
    NewStore->setSyncScopeID(Store->getSyncScopeID());
    
    LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting Store: ====> "
                      << *NewStore << "\n");
    PromotedValues[I] = NewStore;
    Replacements.push_back(Replacement(I, NewStore));
  } else if (isa<LoadInst>(I)) {
    LoadInst *Load = cast<LoadInst>(I);
    
    // Get the pointer operand
    Value *Ptr = Load->getPointerOperand();
    
    // Create a new load instruction
    LoadInst *NewLoad = Builder.CreateLoad(Load->getType(), Ptr, Load->getName());
    
    // Preserve the alignment and other attributes from the original load
    NewLoad->setAlignment(Load->getAlign());
    NewLoad->setVolatile(Load->isVolatile());
    NewLoad->setOrdering(Load->getOrdering());
    NewLoad->setSyncScopeID(Load->getSyncScopeID());
    
    // If the loaded value is of a non-standard type, promote it
    if (auto *IntTy = dyn_cast<IntegerType>(Load->getType())) {
      if (!HipPromoteIntsPass::isStandardBitWidth(IntTy->getBitWidth())) {
        // Promote to standard width
        Type *PromotedLoadType = HipPromoteIntsPass::getPromotedType(Load->getType());
        Value *PromotedValue = nullptr;
        
        if (Load->getType()->getPrimitiveSizeInBits() < PromotedLoadType->getPrimitiveSizeInBits()) {
          LLVM_DEBUG(dbgs() << Indent << "    ZExting loaded value from non-standard type\n");
          PromotedValue = Builder.CreateZExt(NewLoad, PromotedLoadType);
        } else if (Load->getType()->getPrimitiveSizeInBits() > PromotedLoadType->getPrimitiveSizeInBits()) {
          LLVM_DEBUG(dbgs() << Indent << "    Truncing loaded value from non-standard type\n");
          PromotedValue = Builder.CreateTrunc(NewLoad, PromotedLoadType);
        } else {
          LLVM_DEBUG(dbgs() << Indent << "    Bitcasting loaded value from non-standard type\n");
          PromotedValue = Builder.CreateBitCast(NewLoad, PromotedLoadType);
        }
        
        // For non-standard types, we want to use the promoted value internally
        PromotedValues[I] = PromotedValue;
        LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting Load (non-standard): ====> "
                         << *PromotedValue << "\n");
        
        // However, external uses should still see the original type, so we need to
        // replace the original instruction with the non-promoted load
        Replacements.push_back(Replacement(I, NewLoad));
      } else {
        // For standard types, just use the load directly
        PromotedValues[I] = NewLoad;
        LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting Load (standard): ====> "
                         << *NewLoad << "\n");
        Replacements.push_back(Replacement(I, NewLoad));
      }
    } else {
      // Non-integer types are not promoted
      PromotedValues[I] = NewLoad;
      LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting Load (non-int): ====> "
                       << *NewLoad << "\n");
      Replacements.push_back(Replacement(I, NewLoad));
    }
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
      
      LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting Return: ====> "
                        << *NewRet << "\n");
      PromotedValues[I] = NewRet;
      Replacements.push_back(Replacement(I, NewRet));
    } else {
      // Handle void return
      ReturnInst *NewRet = Builder.CreateRetVoid();
      LLVM_DEBUG(dbgs() << Indent << "  " << *I << "   promoting Return: ====> "
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
  // First, check all function signatures for non-standard integer types
  for (Function &F : M) {
    // Check return type
    if (auto *IntTy = dyn_cast<IntegerType>(F.getReturnType())) {
      if (!isStandardBitWidth(IntTy->getBitWidth())) {
        LLVM_DEBUG(dbgs() << "Function " << F.getName() 
                  << " has non-standard integer return type i" 
                  << IntTy->getBitWidth() << ". Aborting.\n");
        return PreservedAnalyses::all();
      }
    }
    
    // Check parameter types
    for (const Argument &Arg : F.args()) {
      if (auto *IntTy = dyn_cast<IntegerType>(Arg.getType())) {
        if (!isStandardBitWidth(IntTy->getBitWidth())) {
          LLVM_DEBUG(dbgs() << "Function " << F.getName() 
                    << " has parameter with non-standard integer type i" 
                    << IntTy->getBitWidth() << ". Aborting.\n");
          return PreservedAnalyses::all();
        }
      }
    }
  }

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

          // Update uses and cleanup
          // First, replace all uses in instructions that are not in our visited set
          for (const auto &R : Replacements) {
            LLVM_DEBUG(dbgs() << "Replacing uses of: " << *R.Old << "\n"
                             << "    with: " << *R.New << "\n");
            // Make a copy of the users to avoid iterator invalidation
            SmallVector<User*, 8> Users(R.Old->user_begin(), R.Old->user_end());
            for (User *U : Users) {
              if (auto *I = dyn_cast<Instruction>(U)) {
                if (!GlobalVisited.count(I) || PromotedValues.count(I)) {
                  LLVM_DEBUG(dbgs() << "  Updating use in: " << *U << "\n");
                  U->replaceUsesOfWith(R.Old, R.New);
                }
              } else {
                // Non-instruction users should be updated as well
                LLVM_DEBUG(dbgs() << "  Updating non-instruction use in: " << *U << "\n");
                U->replaceUsesOfWith(R.Old, R.New);
              }
            }
          }
          
          // Then, for any instructions with remaining uses, we need a different approach
          for (auto &R : Replacements) {
            if (!R.Old->use_empty()) {
              LLVM_DEBUG(dbgs() << "Instruction still has uses after replacement: " << *R.Old << "\n");
              R.Old->replaceAllUsesWith(R.New);
            }
          }

          // Finally, delete the original instructions in reverse order to handle dependencies
          for (auto It = Replacements.rbegin(); It != Replacements.rend(); ++It) {
            LLVM_DEBUG(dbgs() << "Deleting instruction: " << *(It->Old) << "\n");
            if (!It->Old->use_empty()) {
              LLVM_DEBUG(dbgs() << "WARNING: Instruction still has uses before deletion: " << *(It->Old) << "\n");
              // Force replacement again to handle circular references
              It->Old->replaceAllUsesWith(It->New);
            }
            It->Old->eraseFromParent();
          }

          Changed = true;
        }
      }
    }
  }

  // Print the final IR state before exiting
  LLVM_DEBUG(dbgs() << "Final module IR after HipPromoteIntsPass:\n");
  LLVM_DEBUG(M.print(dbgs(), nullptr));

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}