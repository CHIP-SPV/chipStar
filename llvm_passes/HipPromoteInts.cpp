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

// Structure to hold pending PHI node additions
struct PendingPhiAdd {
  PHINode *TargetPhi;
  Value *OriginalValue;
  BasicBlock *IncomingBlock;
};

// Helper to get or create promoted value
static Value *getPromotedValue(Value *V, Type *NonStdType, Type *PromotedTy,
                               IRBuilder<> &Builder, const std::string &Indent,
                               SmallDenseMap<Value *, Value *> &PromotedValues) {
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
  LLVM_DEBUG(dbgs() << Indent << "      Using original value: " << *V << "\n");
  return V;
};

static void processPhiNode(PHINode *Phi, Type *NonStdType, Type *PromotedTy,
                           IRBuilder<> &Builder, const std::string &Indent,
                           SmallVectorImpl<Replacement> &Replacements,
                           SmallDenseMap<Value *, Value *> &PromotedValues,
                           SmallVectorImpl<PendingPhiAdd> &PendingPhiAdds) {
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

  // Copy all incoming values and blocks, potentially deferring some
  for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
    Value *IncomingValue = Phi->getIncomingValue(i);
    BasicBlock *IncomingBlock = Phi->getIncomingBlock(i);

    LLVM_DEBUG(dbgs() << Indent << "    Processing incoming value: " << *IncomingValue
                      << " from block: " << IncomingBlock->getName() << "\n");

    Value *NewIncomingValue = nullptr;

    // Special handling for truncation instructions that convert from standard to non-standard types
    if (auto *TruncI = dyn_cast<TruncInst>(IncomingValue)) {
      Value *TruncSrc = TruncI->getOperand(0);
      Type *SrcTy = TruncSrc->getType();
      Type *DestTy = TruncI->getDestTy();

      bool IsSrcStandard = false;
      if (auto *SrcIntTy = dyn_cast<IntegerType>(SrcTy)) {
        IsSrcStandard = HipPromoteIntsPass::isStandardBitWidth(SrcIntTy->getBitWidth());
      }

      bool IsDestNonStandard = false;
      if (auto *DestIntTy = dyn_cast<IntegerType>(DestTy)) {
        IsDestNonStandard = !HipPromoteIntsPass::isStandardBitWidth(DestIntTy->getBitWidth());
      }

      // Handle truncation from standard to non-standard specially
      if (IsSrcStandard && IsDestNonStandard) {
        LLVM_DEBUG(dbgs() << Indent << "      Found truncation from standard to non-standard type: " << *TruncI << "\n");

        // Check if we already have a promoted value for this truncation
        if (PromotedValues.count(IncomingValue)) {
          NewIncomingValue = PromotedValues[IncomingValue];
          LLVM_DEBUG(dbgs() << Indent << "      Using existing promoted value: " << *NewIncomingValue << "\n");
        } else {
          // If not already processed, get the promoted source directly
          Value *PromotedSrc = PromotedValues.count(TruncSrc) ? PromotedValues[TruncSrc] : TruncSrc;

          // For standard to non-standard truncation, we use the source value directly
          // This makes the truncation effectively a no-op in our promotion chain
          LLVM_DEBUG(dbgs() << Indent << "      Using source directly for standard-to-nonstandard trunc: "
                            << *PromotedSrc << "\n");

          NewIncomingValue = PromotedSrc; // Use the (potentially promoted) source

          // Store this for future use (map the original trunc to the promoted source)
          PromotedValues[IncomingValue] = NewIncomingValue;
        }
      } else if (!IsSrcStandard) {
        // Check if the source operand has a non-standard type
        LLVM_DEBUG(dbgs() << Indent << "      Found truncation from non-standard type: " << *TruncI << "\n");

        // Instead of creating a new truncation chain, we need to handle the chain consistently
        // Check if the truncation itself has already been processed/promoted
        if (PromotedValues.count(IncomingValue)) {
          NewIncomingValue = PromotedValues[IncomingValue];
          LLVM_DEBUG(dbgs() << Indent << "      Using existing promoted value for truncation: "
                           << *NewIncomingValue << "\n");
        } else {
          // If this truncation hasn't been processed yet, it will be handled later by promoteChain.
          // Defer adding it to the PHI.
          LLVM_DEBUG(dbgs() << Indent << "      Deferring PHI add for unprocessed non-standard truncation\n");
          PendingPhiAdds.push_back({NewPhi, IncomingValue, IncomingBlock});
          continue; // Skip the rest of the loop iteration
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
            LLVM_DEBUG(dbgs() << Indent << "      Deferring PHI add for unprocessed non-standard zext\n");
            PendingPhiAdds.push_back({NewPhi, IncomingValue, IncomingBlock});
            continue; // Skip the rest of the loop iteration
          }
        }
      }
    }

    // If not handled by special case above, use normal promotion
    if (!NewIncomingValue) {
      // Check if the incoming value is already promoted
      if (PromotedValues.count(IncomingValue)) {
        NewIncomingValue = PromotedValues[IncomingValue];
         LLVM_DEBUG(dbgs() << Indent << "      Using existing promoted value: " << *NewIncomingValue << "\n");
      } else if (isa<Instruction>(IncomingValue)) {
        // If it's an instruction but not promoted yet, defer it
        LLVM_DEBUG(dbgs() << Indent << "      Deferring PHI add for unprocessed instruction: " << *IncomingValue << "\n");
        PendingPhiAdds.push_back({NewPhi, IncomingValue, IncomingBlock});
        continue; // Skip the rest of the loop iteration
      } else {
        // Must be a constant, argument, or global - use directly
        NewIncomingValue = IncomingValue;
        LLVM_DEBUG(dbgs() << Indent << "      Using non-instruction value directly: " << *NewIncomingValue << "\n");
      }
    }

    // If we have a value (either promoted or original non-instruction), add it now
    // Ensure the type matches the promoted PHI type
    if (NewIncomingValue->getType() != PromotedType) {
      // Use a temporary builder placed before the original PHI
      IRBuilder<> PhiBuilder(Phi);
      if (NewIncomingValue->getType()->getPrimitiveSizeInBits() < PromotedType->getPrimitiveSizeInBits()) {
        LLVM_DEBUG(dbgs() << Indent << "      zExting incoming value: " << *IncomingValue);
        NewIncomingValue = PhiBuilder.CreateZExt(NewIncomingValue, PromotedType);
        LLVM_DEBUG(dbgs() << " ===> " << *NewIncomingValue << "\n");
      } else if (NewIncomingValue->getType()->getPrimitiveSizeInBits() > PromotedType->getPrimitiveSizeInBits()) {
        LLVM_DEBUG(dbgs() << Indent << "      truncing incoming value: " << *IncomingValue);
        NewIncomingValue = PhiBuilder.CreateTrunc(NewIncomingValue, PromotedType);
        LLVM_DEBUG(dbgs() << " ===> " << *NewIncomingValue << "\n");
      } else {
        // Same bit width but different types
        LLVM_DEBUG(dbgs() << Indent << "      bitcasting incoming value: " << *IncomingValue);
        NewIncomingValue = PhiBuilder.CreateBitCast(NewIncomingValue, PromotedType);
        LLVM_DEBUG(dbgs() << " ===> " << *NewIncomingValue << "\n");
      }
    }

    NewPhi->addIncoming(NewIncomingValue, IncomingBlock);
  }

  LLVM_DEBUG(dbgs() << Indent << "  " << *Phi << "   promoting PHI node: ====> " << *NewPhi
                    << "\n");
  Replacements.push_back(Replacement(Phi, NewPhi));
}

static void processZExtInst(ZExtInst *ZExtI, Type *NonStdType, Type *PromotedTy,
                            IRBuilder<> &Builder, const std::string &Indent,
                            SmallVectorImpl<Replacement> &Replacements,
                            SmallDenseMap<Value *, Value *> &PromotedValues) {
  Value *SrcOp = ZExtI->getOperand(0);

  // Get promoted source if available
  Value *PromotedSrc = getPromotedValue(SrcOp, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // Handle zero extension based on source and destination types
  if (SrcOp->getType() == NonStdType) {
    // If the source is our non-standard type, we need to be careful with type consistency
    LLVM_DEBUG(dbgs() << Indent << "  ZExt with non-standard source type: " << *ZExtI << "\n");

    // Create a new zext with properly promoted types
    Type *DestTy = ZExtI->getDestTy();

    // If destination type is smaller than our promoted type, we need to truncate first
    if (DestTy->getPrimitiveSizeInBits() < PromotedTy->getPrimitiveSizeInBits()) {
      Value *NewZExt = Builder.CreateTrunc(PromotedSrc, DestTy);
      LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (with trunc): ====> "
                        << *NewZExt << "\n");
      PromotedValues[ZExtI] = NewZExt;
      Replacements.push_back(Replacement(ZExtI, NewZExt));
    }
    // If destination type is larger, create a proper zext
    else if (DestTy->getPrimitiveSizeInBits() > PromotedTy->getPrimitiveSizeInBits()) {
      Value *NewZExt = Builder.CreateZExt(PromotedSrc, DestTy);
      LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (with zext): ====> "
                        << *NewZExt << "\n");
      PromotedValues[ZExtI] = NewZExt;
      Replacements.push_back(Replacement(ZExtI, NewZExt));
    }
    // If destination is exactly our promoted type, just use the promoted source
    else {
      LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (direct): ====> "
                        << *PromotedSrc << "\n");
      PromotedValues[ZExtI] = PromotedSrc;
      Replacements.push_back(Replacement(ZExtI, PromotedSrc));
    }
  } else {
    // For standard source types, check if destination type is non-standard
    Type *DestTy = ZExtI->getDestTy();

    if (auto *IntTy = dyn_cast<IntegerType>(DestTy)) {
      if (!HipPromoteIntsPass::isStandardBitWidth(IntTy->getBitWidth())) {
        // Promote the destination type to standard width
        Type *PromotedDestTy = HipPromoteIntsPass::getPromotedType(DestTy);

        // Create a direct zext from source to promoted destination type
        Value *NewZExt = Builder.CreateZExt(PromotedSrc, PromotedDestTy);

        LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (non-std dest): ====> "
                       << *NewZExt << "\n");
        PromotedValues[ZExtI] = NewZExt;
        Replacements.push_back(Replacement(ZExtI, NewZExt));
        return;
      }
    }

    // For standard destination types, handle normally
    // Ensure source is promoted if needed *before* casting to dest type
    if (PromotedSrc->getType() != PromotedTy &&
        PromotedSrc->getType()->getPrimitiveSizeInBits() < PromotedTy->getPrimitiveSizeInBits()) {
      PromotedSrc = Builder.CreateZExt(PromotedSrc, PromotedTy);
      LLVM_DEBUG(dbgs() << Indent << "    Implicitly ZExting source for standard ZExt\n");
    }

    // Create a new zext/trunc/noop to the destination type
    Value *NewZExt;

    if (PromotedSrc->getType()->getPrimitiveSizeInBits() > DestTy->getPrimitiveSizeInBits()) {
      NewZExt = Builder.CreateTrunc(PromotedSrc, DestTy);
    } else if (PromotedSrc->getType()->getPrimitiveSizeInBits() < DestTy->getPrimitiveSizeInBits()) {
      NewZExt = Builder.CreateZExt(PromotedSrc, DestTy);
    } else {
      NewZExt = PromotedSrc; // Same size, no conversion needed
    }

    LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt: ====> "
                     << *NewZExt << "\n");
    PromotedValues[ZExtI] = NewZExt;
    Replacements.push_back(Replacement(ZExtI, NewZExt));
  }
}

static void processTruncInst(TruncInst *TruncI, Type *NonStdType, Type *PromotedTy,
                             IRBuilder<> &Builder, const std::string &Indent,
                             SmallVectorImpl<Replacement> &Replacements,
                             SmallDenseMap<Value *, Value *> &PromotedValues) {
  Value *SrcOp = TruncI->getOperand(0);
  Value *PromotedSrc = getPromotedValue(SrcOp, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // Check if we're truncating from a standard type to a non-standard type
  Type *DestTy = TruncI->getDestTy();
  bool IsDestNonStandard = false;
  if (auto *IntTy = dyn_cast<IntegerType>(DestTy)) {
    IsDestNonStandard = !HipPromoteIntsPass::isStandardBitWidth(IntTy->getBitWidth());
  }

  bool IsSrcStandard = false;
  if (auto *SrcIntTy = dyn_cast<IntegerType>(SrcOp->getType())) {
    IsSrcStandard = HipPromoteIntsPass::isStandardBitWidth(SrcIntTy->getBitWidth());
  }

  // Special handling for truncation from standard type to non-standard type
  if (IsSrcStandard && IsDestNonStandard) {
    LLVM_DEBUG(dbgs() << Indent << "  " << *TruncI
              << "   truncation from standard to non-standard becomes no-op: ====> "
              << *PromotedSrc << "\n");

    // For internal use within our promotion chain, we use the source value directly
    // (effectively making the truncation a no-op)
    PromotedValues[TruncI] = PromotedSrc;

    // Replace the original instruction with its (promoted) input operand directly,
    // effectively eliminating the non-standard type production.
    Replacements.push_back(Replacement(TruncI, PromotedSrc));
    return;
  }

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
  LLVM_DEBUG(dbgs() << Indent << "  " << *TruncI << "   promoting Trunc: ====> "
                    << *NewTrunc << "\n");

  // Store both the promoted and truncated versions
  PromotedValues[TruncI] = NewTrunc; // Use truncated version as the default

  // Only use the promoted version for internal operations on non-standard types
  if (auto *IntTy = dyn_cast<IntegerType>(TruncI->getDestTy())) {
    if (!HipPromoteIntsPass::isStandardBitWidth(IntTy->getBitWidth())) {
      // For non-standard destination types, store the promoted version for our internal use
      PromotedValues[TruncI] = PromotedSrc;
    }
  }

  Replacements.push_back(Replacement(TruncI, NewTrunc));
}

static void processBinaryOperator(BinaryOperator *BinOp, Type *NonStdType, Type *PromotedTy,
                                  IRBuilder<> &Builder, const std::string &Indent,
                                  SmallVectorImpl<Replacement> &Replacements,
                                  SmallDenseMap<Value *, Value *> &PromotedValues) {
  bool NeedsPromotion = (BinOp->getType() == NonStdType);

  Value *LHS = getPromotedValue(BinOp->getOperand(0), NonStdType, PromotedTy, Builder, Indent, PromotedValues);
  Value *RHS = getPromotedValue(BinOp->getOperand(1), NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  Value *NewInst;
  if (NeedsPromotion) {
    // Create operation in promoted type
    LLVM_DEBUG(dbgs() << Indent << "  Creating a binary operation Opcode: " << BinOp->getOpcodeName() << " LHS: " << *LHS << " and RHS: " << *RHS << "\n");
    NewInst = Builder.CreateBinOp(BinOp->getOpcode(), LHS, RHS);
  } else {
    // For operations that should stay in original type
    Type* OriginalType = BinOp->getType();
    if (LHS->getType() != OriginalType) {
        if (LHS->getType()->getPrimitiveSizeInBits() > OriginalType->getPrimitiveSizeInBits())
            LHS = Builder.CreateTrunc(LHS, OriginalType);
        else if (LHS->getType()->getPrimitiveSizeInBits() < OriginalType->getPrimitiveSizeInBits())
             LHS = Builder.CreateZExt(LHS, OriginalType);
        else
             LHS = Builder.CreateBitCast(LHS, OriginalType);
    }
    if (RHS->getType() != OriginalType) {
         if (RHS->getType()->getPrimitiveSizeInBits() > OriginalType->getPrimitiveSizeInBits())
            RHS = Builder.CreateTrunc(RHS, OriginalType);
        else if (RHS->getType()->getPrimitiveSizeInBits() < OriginalType->getPrimitiveSizeInBits())
             RHS = Builder.CreateZExt(RHS, OriginalType);
        else
             RHS = Builder.CreateBitCast(RHS, OriginalType);
    }
    NewInst = Builder.CreateBinOp(BinOp->getOpcode(), LHS, RHS);
  }

  LLVM_DEBUG(dbgs() << Indent << "  " << *BinOp << "   promoting BinOp: ====> "
                    << *NewInst << "\n");
  PromotedValues[BinOp] = NewInst;
  Replacements.push_back(Replacement(BinOp, NewInst));
}

static void processSelectInst(SelectInst *SelI, Type *NonStdType, Type *PromotedTy,
                              IRBuilder<> &Builder, const std::string &Indent,
                              SmallVectorImpl<Replacement> &Replacements,
                              SmallDenseMap<Value *, Value *> &PromotedValues) {
  bool NeedsPromotion = (SelI->getType() == NonStdType);

  // Get promoted operands
  Value *Condition = getPromotedValue(SelI->getCondition(), NonStdType, PromotedTy, Builder, Indent, PromotedValues);
  Value *TrueVal = getPromotedValue(SelI->getTrueValue(), NonStdType, PromotedTy, Builder, Indent, PromotedValues);
  Value *FalseVal = getPromotedValue(SelI->getFalseValue(), NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // Make sure condition is i1
  if (Condition->getType() != Type::getInt1Ty(SelI->getContext())) {
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
    auto adjustType = [&](Value *V, const std::string& val_name) -> Value* {
        if (V->getType() != OriginalType) {
            if (V->getType()->getPrimitiveSizeInBits() > OriginalType->getPrimitiveSizeInBits()) {
                V = Builder.CreateTrunc(V, OriginalType);
                LLVM_DEBUG(dbgs() << Indent << "    Truncating " << val_name << " value to match select type\n");
            } else if (V->getType()->getPrimitiveSizeInBits() < OriginalType->getPrimitiveSizeInBits()) {
                V = Builder.CreateZExt(V, OriginalType);
                LLVM_DEBUG(dbgs() << Indent << "    Extending " << val_name << " value to match select type\n");
            } else {
                V = Builder.CreateBitCast(V, OriginalType);
                LLVM_DEBUG(dbgs() << Indent << "    Bitcasting " << val_name << " value to match select type\n");
            }
        }
        return V;
    };
    TrueVal = adjustType(TrueVal, "true");
    FalseVal = adjustType(FalseVal, "false");

    NewSelect = Builder.CreateSelect(Condition, TrueVal, FalseVal, SelI->getName());
  }

  LLVM_DEBUG(dbgs() << Indent << "  " << *SelI << "   promoting Select: ====> "
                    << *NewSelect << "\n");
  PromotedValues[SelI] = NewSelect;
  Replacements.push_back(Replacement(SelI, NewSelect));
}

static void processICmpInst(ICmpInst *CmpI, Type *NonStdType, Type *PromotedTy,
                            IRBuilder<> &Builder, const std::string &Indent,
                            SmallVectorImpl<Replacement> &Replacements,
                            SmallDenseMap<Value *, Value *> &PromotedValues) {
  // Get promoted operands
  Value *LHS = getPromotedValue(CmpI->getOperand(0), NonStdType, PromotedTy, Builder, Indent, PromotedValues);
  Value *RHS = getPromotedValue(CmpI->getOperand(1), NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // Make sure operands are of same type for comparison
  if (LHS->getType() != RHS->getType()) {
    // Determine the common type (prefer promoted type if involved)
    Type *CommonType = nullptr;
    if (LHS->getType() == PromotedTy || RHS->getType() == PromotedTy) {
        CommonType = PromotedTy;
    } else {
        // If neither is promoted type but they still differ, convert to largest type
        CommonType = LHS->getType()->getPrimitiveSizeInBits() >
                     RHS->getType()->getPrimitiveSizeInBits() ?
                     LHS->getType() : RHS->getType();
    }

    // Cast LHS if needed
    if (LHS->getType() != CommonType) {
        if (LHS->getType()->getPrimitiveSizeInBits() < CommonType->getPrimitiveSizeInBits())
            LHS = Builder.CreateZExt(LHS, CommonType);
        else if (LHS->getType()->getPrimitiveSizeInBits() > CommonType->getPrimitiveSizeInBits())
            LHS = Builder.CreateTrunc(LHS, CommonType);
        else
            LHS = Builder.CreateBitCast(LHS, CommonType);
        LLVM_DEBUG(dbgs() << Indent << "    Adjusting ICmp LHS type to " << *CommonType << "\n");
    }
    // Cast RHS if needed
    if (RHS->getType() != CommonType) {
         if (RHS->getType()->getPrimitiveSizeInBits() < CommonType->getPrimitiveSizeInBits())
            RHS = Builder.CreateZExt(RHS, CommonType);
        else if (RHS->getType()->getPrimitiveSizeInBits() > CommonType->getPrimitiveSizeInBits())
            RHS = Builder.CreateTrunc(RHS, CommonType);
        else
            RHS = Builder.CreateBitCast(RHS, CommonType);
        LLVM_DEBUG(dbgs() << Indent << "    Adjusting ICmp RHS type to " << *CommonType << "\n");
    }
  }

  // Create new comparison instruction
  Value *NewCmp = Builder.CreateICmp(CmpI->getPredicate(), LHS, RHS, CmpI->getName());

  LLVM_DEBUG(dbgs() << Indent << "  " << *CmpI << "   promoting ICmp: ====> "
                    << *NewCmp << "\n");
  PromotedValues[CmpI] = NewCmp;
  Replacements.push_back(Replacement(CmpI, NewCmp));
}

static void processCallInst(CallInst *OldCall, Type *NonStdType, Type *PromotedTy,
                            IRBuilder<> &Builder, const std::string &Indent,
                            SmallVectorImpl<Replacement> &Replacements,
                            SmallDenseMap<Value *, Value *> &PromotedValues) {
  // Create a new call with the same operands, but use promoted values where
  // available
  SmallVector<Value *, 8> NewArgs;
  for (unsigned i = 0; i < OldCall->arg_size(); ++i) {
    Value *OldArg = OldCall->getArgOperand(i);
    Value *NewArg = getPromotedValue(OldArg, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

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
            LLVM_DEBUG(dbgs() << Indent << "    Truncating argument " << i << " from "
                     << *ArgIntTy << " to " << *IntTy << "\n");
          }
          // If expected type is larger, extend
          else if (IntTy->getBitWidth() > ArgIntTy->getBitWidth()) {
            NewArg = Builder.CreateZExt(NewArg, ExpectedType);
            LLVM_DEBUG(dbgs() << Indent << "    Extending argument " << i << " from "
                     << *ArgIntTy << " to " << *IntTy << "\n");
          } else {
            NewArg = Builder.CreateBitCast(NewArg, ExpectedType);
            LLVM_DEBUG(dbgs() << Indent << "    Bitcasting argument " << i << " to " << *IntTy << "\n");
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

  LLVM_DEBUG(dbgs() << Indent << "  " << *OldCall << "   promoting Call: ====> "
                    << *NewCall << "\n");
  PromotedValues[OldCall] = NewCall;
  Replacements.push_back(Replacement(OldCall, NewCall));
}

static void processStoreInst(StoreInst *Store, Type *NonStdType, Type *PromotedTy,
                             IRBuilder<> &Builder, const std::string &Indent,
                             SmallVectorImpl<Replacement> &Replacements,
                             SmallDenseMap<Value *, Value *> &PromotedValues) {
  // Get the value being stored (possibly promoted)
  Value *StoredValue = Store->getValueOperand();
  Value *NewStoredValue = getPromotedValue(StoredValue, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

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

  LLVM_DEBUG(dbgs() << Indent << "  " << *Store << "   promoting Store: ====> "
                    << *NewStore << "\n");
  // Store instructions don't produce a value, so we don't put them in PromotedValues
  Replacements.push_back(Replacement(Store, NewStore));
}


static void processLoadInst(LoadInst *Load, Type *NonStdType, Type *PromotedTy,
                            IRBuilder<> &Builder, const std::string &Indent,
                            SmallVectorImpl<Replacement> &Replacements,
                            SmallDenseMap<Value *, Value *> &PromotedValues) {
   // Get the pointer operand
  Value *Ptr = Load->getPointerOperand();

  // Create a new load instruction with the original type
  LoadInst *NewLoad = Builder.CreateLoad(Load->getType(), Ptr, Load->getName());

  // Preserve the alignment and other attributes from the original load
  NewLoad->setAlignment(Load->getAlign());
  NewLoad->setVolatile(Load->isVolatile());
  NewLoad->setOrdering(Load->getOrdering());
  NewLoad->setSyncScopeID(Load->getSyncScopeID());

  // Now, check if the loaded value is of a non-standard type that needs promotion
  Value *ResultValue = NewLoad; // Default to the newly created load
  if (auto *IntTy = dyn_cast<IntegerType>(Load->getType())) {
    if (!HipPromoteIntsPass::isStandardBitWidth(IntTy->getBitWidth())) {
      // Promote to standard width
      Type *PromotedLoadType = HipPromoteIntsPass::getPromotedType(Load->getType());
      Value *PromotedValue = nullptr;

      // Use a temporary builder positioned *after* the NewLoad
      IRBuilder<> AfterLoadBuilder(NewLoad->getNextNode());

      if (Load->getType()->getPrimitiveSizeInBits() < PromotedLoadType->getPrimitiveSizeInBits()) {
        LLVM_DEBUG(dbgs() << Indent << "    ZExting loaded value from non-standard type\n");
        PromotedValue = AfterLoadBuilder.CreateZExt(NewLoad, PromotedLoadType, Load->getName() + ".promoted");
      } else if (Load->getType()->getPrimitiveSizeInBits() > PromotedLoadType->getPrimitiveSizeInBits()) {
        LLVM_DEBUG(dbgs() << Indent << "    Truncing loaded value from non-standard type\n");
        PromotedValue = AfterLoadBuilder.CreateTrunc(NewLoad, PromotedLoadType, Load->getName() + ".promoted");
      } else {
        LLVM_DEBUG(dbgs() << Indent << "    Bitcasting loaded value from non-standard type\n");
        PromotedValue = AfterLoadBuilder.CreateBitCast(NewLoad, PromotedLoadType, Load->getName() + ".promoted");
      }

      // For non-standard types, we want to use the promoted value internally
      ResultValue = PromotedValue;
      LLVM_DEBUG(dbgs() << Indent << "  " << *Load << "   promoting Load (non-standard): ====> "
                       << *ResultValue << " (via " << *NewLoad << ")\n");
    } else {
        LLVM_DEBUG(dbgs() << Indent << "  " << *Load << "   promoting Load (standard): ====> "
                         << *ResultValue << "\n");
    }
  } else {
    // Non-integer types are not promoted
    LLVM_DEBUG(dbgs() << Indent << "  " << *Load << "   promoting Load (non-int): ====> "
                     << *ResultValue << "\n");
  }

  // Map the original load instruction to the potentially promoted value for internal use
  PromotedValues[Load] = ResultValue;
  // Replace the original load instruction with the new load instruction (which has the original type)
  Replacements.push_back(Replacement(Load, NewLoad));
}

static void processReturnInst(ReturnInst *RetI, Type *NonStdType, Type *PromotedTy,
                              IRBuilder<> &Builder, const std::string &Indent,
                              SmallVectorImpl<Replacement> &Replacements,
                              SmallDenseMap<Value *, Value *> &PromotedValues) {
   // If there's a return value, check if it needs to be promoted/adjusted
  if (RetI->getNumOperands() > 0) {
    Value *RetVal = RetI->getReturnValue();
    Value *NewRetVal = getPromotedValue(RetVal, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

    // Make sure the return value matches the function's return type
    Type *FuncRetType = RetI->getFunction()->getReturnType();
    if (NewRetVal->getType() != FuncRetType) {
      // If the function return type is larger than the value type, extend it
      if (NewRetVal->getType()->getPrimitiveSizeInBits() < FuncRetType->getPrimitiveSizeInBits()) {
          LLVM_DEBUG(dbgs() << Indent << "    Extending return value for function type\n");
          NewRetVal = Builder.CreateZExt(NewRetVal, FuncRetType);
      }
      // If the function return type is smaller, truncate it
      else if (NewRetVal->getType()->getPrimitiveSizeInBits() > FuncRetType->getPrimitiveSizeInBits()) {
           LLVM_DEBUG(dbgs() << Indent << "    Truncating return value for function type\n");
           NewRetVal = Builder.CreateTrunc(NewRetVal, FuncRetType);
      } else {
           LLVM_DEBUG(dbgs() << Indent << "    Bitcasting return value for function type\n");
           NewRetVal = Builder.CreateBitCast(NewRetVal, FuncRetType);
      }
    }

    // Create a new return instruction with the correctly typed value
    ReturnInst *NewRet = Builder.CreateRet(NewRetVal);

    LLVM_DEBUG(dbgs() << Indent << "  " << *RetI << "   promoting Return: ====> "
                      << *NewRet << "\n");
    // Return instructions don't produce a value, so we don't put them in PromotedValues
    Replacements.push_back(Replacement(RetI, NewRet));
  } else {
    // Handle void return
    ReturnInst *NewRet = Builder.CreateRetVoid();
    LLVM_DEBUG(dbgs() << Indent << "  " << *RetI << "   promoting Return: ====> "
                      << *NewRet << "\n");
    // Return instructions don't produce a value, so we don't put them in PromotedValues
    Replacements.push_back(Replacement(RetI, NewRet));
  }
}

void processInstruction(Instruction *I, Type *NonStdType, Type *PromotedTy,
                        const std::string &Indent,
                        SmallVectorImpl<Replacement> &Replacements,
                        SmallDenseMap<Value *, Value *> &PromotedValues,
                        SmallVectorImpl<PendingPhiAdd> &PendingPhiAdds) {
  IRBuilder<> Builder(I);

  // Dispatch to the appropriate handler based on instruction type
  if (auto *Phi = dyn_cast<PHINode>(I)) {
      processPhiNode(Phi, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues, PendingPhiAdds);
  } else if (auto *ZExtI = dyn_cast<ZExtInst>(I)) {
      processZExtInst(ZExtI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *TruncI = dyn_cast<TruncInst>(I)) {
      processTruncInst(TruncI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *BinOp = dyn_cast<BinaryOperator>(I)) {
      processBinaryOperator(BinOp, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *SelI = dyn_cast<SelectInst>(I)) {
      processSelectInst(SelI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *CmpI = dyn_cast<ICmpInst>(I)) {
      processICmpInst(CmpI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *OldCall = dyn_cast<CallInst>(I)) {
      processCallInst(OldCall, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *Store = dyn_cast<StoreInst>(I)) {
      processStoreInst(Store, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *Load = dyn_cast<LoadInst>(I)) {
      processLoadInst(Load, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *RetI = dyn_cast<ReturnInst>(I)) {
      processReturnInst(RetI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else {
    LLVM_DEBUG(dbgs() << Indent << "  Unhandled instruction type: " << *I << "\n");
    assert(false && "HipPromoteIntsPass: Unhandled instruction type");
  }
}

static void promoteChain(Instruction *OldI, Type *NonStdType, Type *PromotedTy,
                         SmallPtrSetImpl<Instruction *> &Visited,
                         SmallVectorImpl<Replacement> &Replacements,
                         SmallDenseMap<Value *, Value *> &PromotedValues,
                         SmallVectorImpl<PendingPhiAdd> &PendingPhiAdds,
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
                     PromotedValues, PendingPhiAdds);

  // Recursively process all users
  for (User *U : OldI->users())
    if (auto *UI = dyn_cast<Instruction>(U))
      promoteChain(UI, NonStdType, PromotedTy, Visited, Replacements,
                   PromotedValues, PendingPhiAdds, Depth + 1);

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
          SmallVector<PendingPhiAdd, 16> PendingPhiAdds;

          // Use GlobalVisited instead of creating a new set
          promoteChain(I, IntTy, PromotedType, GlobalVisited, Replacements,
                       PromotedValues, PendingPhiAdds, 0);

          // Process pending PHI additions *before* main replacement
          LLVM_DEBUG(dbgs() << "Processing " << PendingPhiAdds.size() << " pending PHI additions...\n");
          for (const auto &Pending : PendingPhiAdds) {
              PHINode *TargetPhi = Pending.TargetPhi;
              Value *OriginalValue = Pending.OriginalValue;
              BasicBlock *IncomingBlock = Pending.IncomingBlock;

              LLVM_DEBUG(dbgs() << "  Pending: Add " << *OriginalValue << " to " << *TargetPhi << " from block " << IncomingBlock->getName() << "\n");

              // The original value should now be in PromotedValues
              assert(PromotedValues.count(OriginalValue) && "Pending PHI value was not promoted!");
              Value *PromotedValue = PromotedValues[OriginalValue];
              LLVM_DEBUG(dbgs() << "    Found promoted value: " << *PromotedValue << "\n");

              // Adjust type if necessary, inserting before the PHI node
              if (PromotedValue->getType() != TargetPhi->getType()) {
                IRBuilder<> PhiBuilder(TargetPhi); // Place instructions before the PHI
                if (PromotedValue->getType()->getPrimitiveSizeInBits() < TargetPhi->getType()->getPrimitiveSizeInBits()) {
                  LLVM_DEBUG(dbgs() << "    Adjusting type (zext) for PHI add\n");
                  PromotedValue = PhiBuilder.CreateZExt(PromotedValue, TargetPhi->getType());
                } else if (PromotedValue->getType()->getPrimitiveSizeInBits() > TargetPhi->getType()->getPrimitiveSizeInBits()) {
                  LLVM_DEBUG(dbgs() << "    Adjusting type (trunc) for PHI add\n");
                  PromotedValue = PhiBuilder.CreateTrunc(PromotedValue, TargetPhi->getType());
                } else {
                  LLVM_DEBUG(dbgs() << "    Adjusting type (bitcast) for PHI add\n");
                  PromotedValue = PhiBuilder.CreateBitCast(PromotedValue, TargetPhi->getType());
                }
                LLVM_DEBUG(dbgs() << "      ==> Adjusted value: " << *PromotedValue << "\n");
              }

              TargetPhi->addIncoming(PromotedValue, IncomingBlock);
          }

          // Now perform the main replacements
          LLVM_DEBUG(dbgs() << "Performing main instruction replacements...\n");
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