#include "HipPromoteInts.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "hip-promote-ints"

/**
 * @brief Promotes non-standard integer types (e.g., i33, i56) to the next standard width.
 *
 * LLVM's loop optimizations can generate integer types with bit widths that are
 * not powers of two (or 1). These non-standard types can cause issues during
 * later stages, particularly SPIR-V translation.
 *
 * Algorithm Overview:
 * ------------------
 * This pass replaces instructions involving non-standard integer types with
 * equivalent instructions using the next largest standard integer type (8, 16, 32, 64).
 * It operates in two main phases within the `run` method for each function:
 *
 * 1. Worklist Collection:
 *    - Identify all instructions that either produce a non-standard integer type
 *      or use an operand with a non-standard integer type.
 *    - These instructions form the starting points for promotion chains.
 *
 * 2. Chain Promotion and Replacement:
 *    - For each instruction in the worklist (that hasn't already been processed):
 *        a. **Recursive Promotion (`promoteChain`)**: Recursively traverse the use-def chain
 *           starting from the initial instruction. For each instruction encountered:
 *           - Determine its corresponding promoted instruction (using standard types).
 *           - Store the mapping from the original instruction to its promoted counterpart
 *             (using `PromotedValues` map).
 *           - Add the original instruction and its replacement to a `Replacements` list.
 *           - Handle type adjustments (zext/trunc/bitcast) as needed to maintain consistency
 *             within the *new* chain being built (which uses promoted types).
 *           - Special handling for PHI nodes is required to manage dependencies correctly,
 *             deferring the addition of incoming values until the producing instruction is processed.
 *        b. **PHI Node Patching**: After processing a chain, resolve any pending PHI node additions.
 *           Look up the promoted values for the original incoming values and add them to the new PHI nodes,
 *           inserting type adjustments if necessary.
 *        c. **Instruction Replacement**: Iterate through the `Replacements` list.
 *           - Replace all uses of the original instruction `Old` with the new value `New`,
 *             *only* updating users that were also part of the processed chain (`GlobalVisited`).
 *           - Perform a final `replaceAllUsesWith` to catch any remaining uses (e.g., external users
 *             or complex dependencies).
 *           - Erase the original instruction `Old`.
 *
 * Key Data Structures:
 * --------------------
 * - `WorkList`: Stores initial instructions needing promotion.
 * - `GlobalVisited`: Tracks instructions already processed across all chains to avoid redundant work.
 * - `PromotedValues`: Maps original `Value*` (instructions, constants) to their promoted `Value*` equivalents
 *   within the context of a single `promoteChain` call.
 * - `Replacements`: Stores pairs of `{original instruction, new value}` created during `promoteChain`.
 * - `PendingPhiAdds`: Temporarily stores information needed to update PHI nodes after their dependencies are processed.
 *
 * Example:
 * --------
 * Original IR:
 *   %1 = zext i32 %x to i33
 *   %2 = add i33 %1, 1
 *   %3 = trunc i33 %2 to i8
 *
 * After Promotion (Conceptual):
 *   %1_promoted = zext i32 %x to i64 ; Handled by processZExtInst (non-std dest)
 *   %const1_promoted = i64 1
 *   %2_promoted = add i64 %1_promoted, %const1_promoted ; Handled by processBinaryOperator
 *   %3_final = trunc i64 %2_promoted to i8 ; Handled by processTruncInst
 *
 * The pass builds the promoted instructions (%1_promoted, %2_promoted, %3_final)
 * alongside the originals, then replaces uses of %1 with %1_promoted (if used later),
 * replaces uses of %2 with %2_promoted, replaces uses of %3 with %3_final, and
 * finally deletes %1, %2, and %3.
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
// TODO: explain when and on what this is called
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

  // Handle Constants specifically
  // Example:
  //   %c = add i33 %a, 1
  // Becomes:
  //   %c_promoted = add i64 %a_promoted, 1
  if (auto *ConstInt = dyn_cast<ConstantInt>(V)) {
    if (ConstInt->getType() == NonStdType) {
      // Create a new ConstantInt with the promoted type
      // zext resolves at compile time, it doesn't generate zext instructions
      APInt PromotedValue = ConstInt->getValue().zext(PromotedTy->getIntegerBitWidth());
      Value *NewConst = ConstantInt::get(PromotedTy, PromotedValue);
      LLVM_DEBUG(dbgs() << Indent << "      Promoting ConstantInt: " << *V
                        << " to " << *NewConst << "\n");
      PromotedValues[V] = NewConst; // Cache the promoted constant
      return NewConst;
    }
    // If it's a constant of a different standard type, it might need zext/trunc later
    // but we return the original constant for now. Adjustments happen in the instruction processing.
     LLVM_DEBUG(dbgs() << Indent << "      Using original ConstantInt: " << *V << "\n");
     return V; // Return original standard-type constant
  }


  // If it's the non-standard type (and not a constant, handled above), promote it
  // Example:
  //   %res = add i33 %a, %b
  // If %res (i33) is used later where an i64 is expected by a promoted instruction:
  //   getPromotedValue(%res, i33, i64, ...) -> creates '%res.zext = zext i33 %res to i64'
  // This %res.zext is then used by the promoted instruction.
  // TODO: %res.zext = zext i33 %res to i64 still contains i33, explain how this is handled
  if (V->getType() == NonStdType) {
    // Check if it's an instruction that should have been processed already
    // This might indicate a circular dependency or an issue in the traversal order.
    // TODO: explain where a circular dependency might happen and give an example.
    assert(isa<Instruction>(V) && !PromotedValues.count(V) && "Encountered unprocessed non-standard instruction");

    Value *NewV = nullptr;
    if (V->getType()->getPrimitiveSizeInBits() < PromotedTy->getPrimitiveSizeInBits()) {
      NewV = Builder.CreateZExt(V, PromotedTy);
      LLVM_DEBUG(dbgs() << Indent << "      Promoting non-standard type with zext: " << *V
                        << " to " << *NewV << "\n");
    } else if (V->getType()->getPrimitiveSizeInBits() > PromotedTy->getPrimitiveSizeInBits()) {
      // TODO: this should never happen, correct?
      NewV = Builder.CreateTrunc(V, PromotedTy);
      LLVM_DEBUG(dbgs() << Indent << "      Promoting non-standard type with trunc: " << *V
                        << " to " << *NewV << "\n");
    } else {
      // TODO: explain why we need to bitcast
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
    // Example:
    // bb1:
    //   %x = add i64 ...
    //   %trunc_std_nonstd = trunc i64 %x to i33
    //   br label %bb2
    // bb2:
    //   %phi = phi i33 [ %trunc_std_nonstd, %bb1 ], ...
    //
    // When processing %phi, we encounter %trunc_std_nonstd.
    // Since it's standard -> non-standard, we directly use the *source* (%x)
    // as the incoming value for the *promoted* PHI node (%phi_promoted of type i64).
    // This effectively bypasses the truncation within the promoted chain.
    //
    // TODO: Explain why we can't use processZExtInst/processTruncInst here
    // We handle this case *within* processPhiNode because we need to determine the
    // correct *incoming value* for the *new* PHI node based on the original incoming
    // value's nature. If the original incoming value is a `trunc` from a standard
    // type, we want the *source* of that `trunc` (which is already standard or will
    // be promoted) to be the incoming value for the new PHI. Calling
    // `processTruncInst` recursively might create unnecessary intermediate instructions
    // or lead to incorrect handling if the `TruncInst` hasn't been visited yet by the
    // main `promoteChain` traversal. This special handling simplifies the PHI logic.
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
      // e.x. %trunc_std_nonstd = trunc i64 %x to i33
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
    // TODO: IR Example
    // Example:
    // bb1:
    //   %y = add i33 ...
    //   %zext_nonstd_std = zext i33 %y to i64
    //   br label %bb2
    // bb2:
    //   %phi = phi i64 [ %zext_nonstd_std, %bb1 ], ...
    //
    // When processing %phi, we encounter %zext_nonstd_std.
    // If %zext_nonstd_std has already been processed by promoteChain,
    // PromotedValues[%zext_nonstd_std] will exist (and be the correct i64 value),
    // so we use that.
    // If it hasn't been processed yet, we defer adding it to the PHI using
    // PendingPhiAdds. promoteChain will eventually process %zext_nonstd_std,
    // create its replacement, and then the pending add will be resolved.
    //
    // TODO: Explain why we can't use processZExtInst/processTruncInst here
    // Similar to the `trunc` case, we defer if the source instruction (`ZExtInst`)
    // hasn't been processed yet by `promoteChain`. This ensures that when we
    // add the incoming value to the new PHI, we use the *result* of the already-promoted
    // `ZExtInst`. Calling `processZExtInst` recursively here could disrupt the
    // planned processing order managed by `promoteChain` and `GlobalVisited`.
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
    // TODO: IR Example
    // Example:
    // bb1:
    //   %const = i33 42
    //   br label %bb2
    // bb2:
    //   %phi = phi i33 [ %const, %bb1 ], ...
    //
    // When processing %phi (type i33), the incoming %const (i33) is not an instruction.
    // getPromotedValue will handle promoting the constant to i64.
    // This promoted constant (i64 42) will be used as NewIncomingValue.
    // If %phi's promoted type is i64, no further adjustment is needed.
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
    // TODO: IR Example
    // Example:
    //   %phi = phi i33 [ %incoming_i8, %bb1 ], ...
    // Assume %phi promotes to %phi_promoted (i64).
    // NewIncomingValue might be %incoming_i8 (original if not promoted elsewhere) or
    // a promoted version if it came from another processed instruction.
    // Let NewIncomingValue be %val (type i8).
    // Since %val (i8) != %phi_promoted (i64), we need adjustment.
    // This block will insert:
    //   %zext_for_phi = zext i8 %val to i64 (inserted before %phi)
    // Then, %zext_for_phi will be added as the incoming value to %phi_promoted.
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

// Helper to check for non-standard integer types
static bool isNonStandardInt(Type *T) {
  if (auto *IntTy = dyn_cast<IntegerType>(T)) {
    return !HipPromoteIntsPass::isStandardBitWidth(IntTy->getBitWidth());
  }
  return false;
}

// Helper to adjust value V to type TargetTy using Builder
static Value* adjustType(Value *V, Type *TargetTy, IRBuilder<> &Builder, const std::string& Indent = "") {
    if (V->getType() == TargetTy) {
        return V;
    }

    // TODO: IR Example
    // Example 1: Source i32, Target i64
    //   adjustType(%val_i32, i64, builder) -> creates '%zext = zext i32 %val_i32 to i64'
    // Example 2: Source i64, Target i32
    //   adjustType(%val_i64, i32, builder) -> creates '%trunc = trunc i64 %val_i64 to i32'
    // Example 3: Source i64, Target <64 x i1> (Different types, same size)
    //   adjustType(%val_i64, <64 x i1>, builder) -> creates '%bitcast = bitcast i64 %val_i64 to <64 x i1>'

    unsigned SrcBits = V->getType()->getPrimitiveSizeInBits();
    unsigned DstBits = TargetTy->getPrimitiveSizeInBits();

    LLVM_DEBUG(dbgs() << Indent << "Adjusting type of " << *V << " from " << *V->getType() << " to " << *TargetTy << "\n");

    Value* AdjustedV = nullptr;
    if (DstBits < SrcBits) {
        AdjustedV = Builder.CreateTrunc(V, TargetTy);
         LLVM_DEBUG(dbgs() << Indent << "  Created Trunc: " << *AdjustedV << "\n");
    } else if (DstBits > SrcBits) {
        AdjustedV = Builder.CreateZExt(V, TargetTy);
         LLVM_DEBUG(dbgs() << Indent << "  Created ZExt: " << *AdjustedV << "\n");
    } else {
        AdjustedV = Builder.CreateBitCast(V, TargetTy);
         LLVM_DEBUG(dbgs() << Indent << "  Created BitCast: " << *AdjustedV << "\n");
    }
    return AdjustedV;
}

// Refined processZExtInst logic:
static void processZExtInst(ZExtInst *ZExtI, Type *NonStdType /* Type being promoted, e.g. i56 */,
                            Type *PromotedTy /* Type to promote to, e.g. i64 */,
                            IRBuilder<> &Builder, const std::string &Indent,
                            SmallVectorImpl<Replacement> &Replacements,
                            SmallDenseMap<Value *, Value *> &PromotedValues) {
  Value *SrcOp = ZExtI->getOperand(0);
  Type *SrcTy = SrcOp->getType();
  Type *DestTy = ZExtI->getDestTy();

  // Get the potentially promoted source value. getPromotedValue handles constant promotion.
  Value *PromotedSrc = getPromotedValue(SrcOp, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  bool IsSrcNonStandard = isNonStandardInt(SrcTy);
  bool IsDestNonStandard = isNonStandardInt(DestTy);

  Value *NewValue = nullptr;

  if (IsSrcNonStandard) {
    // Case 1: Source is Non-Standard (e.g., zext i56 -> i64)
    // PromotedSrc should now have the PromotedTy (e.g., i64)
    assert(PromotedSrc->getType() == PromotedTy && "Non-standard source operand was not promoted correctly in getPromotedValue");

    // We need the final result type to be DestTy (original destination type).
    // Adjust the PromotedSrc to match DestTy if necessary.
    NewValue = adjustType(PromotedSrc, DestTy, Builder, Indent + "    ");
    if (NewValue == PromotedSrc) {
      LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (non-std src, becomes no-op): ====> " << *NewValue << "\n");
    } else {
      LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (non-std src, requires adjust): ====> " << *NewValue << "\n");
    }

    // Replace the original ZExt instruction with the NewValue.
    PromotedValues[ZExtI] = NewValue; // Map original ZExt result to the final value
    Replacements.push_back(Replacement(ZExtI, NewValue));

  } else if (IsDestNonStandard) {
    // Case 2: Destination is Non-Standard (e.g., zext i32 -> i56)
    Type *PromotedDestTy = HipPromoteIntsPass::getPromotedType(DestTy); // e.g., i64

    // Ensure the source operand used for the new ZExt has its original standard type.
    Value* AdjustedSrc = PromotedSrc;
    if (AdjustedSrc->getType() != SrcTy) {
        IRBuilder<> TmpBuilder(ZExtI); // Builder before original instruction
        LLVM_DEBUG(dbgs() << Indent << "    Adjusting source back to original standard type (" << *SrcTy << ") for zext->non-std\n");
        AdjustedSrc = adjustType(AdjustedSrc, SrcTy, TmpBuilder, Indent + "      ");
    }

    // Create ZExt using the adjusted standard source to the *promoted* destination type.
    NewValue = Builder.CreateZExt(AdjustedSrc, PromotedDestTy);
    LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (non-std dest): ====> " << *NewValue << "\n");
    PromotedValues[ZExtI] = NewValue; // Map original ZExt result to the new promoted value
    Replacements.push_back(Replacement(ZExtI, NewValue));

  } else {
    // Case 3: Source and Destination are Standard (e.g., zext i32 -> i64)
    // Recreate the instruction, ensuring the source operand has its original standard type.
     Value* AdjustedSrc = PromotedSrc;
     if (AdjustedSrc->getType() != SrcTy) {
        IRBuilder<> TmpBuilder(ZExtI); // Builder before original instruction
        LLVM_DEBUG(dbgs() << Indent << "    Adjusting source back to original standard type (" << *SrcTy << ") for std zext\n");
        AdjustedSrc = adjustType(AdjustedSrc, SrcTy, TmpBuilder, Indent + "      ");
     }

    // Create the ZExt with the adjusted standard source and original standard destination type.
    NewValue = Builder.CreateZExt(AdjustedSrc, DestTy);
    LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (standard src/dest): ====> " << *NewValue << "\n");
    // Map original ZExt result to new ZExt result. Both should have same standard type.
    PromotedValues[ZExtI] = NewValue;
    Replacements.push_back(Replacement(ZExtI, NewValue));
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
  bool NeedsPromotion = isNonStandardInt(BinOp->getType());

  Value *LHS = getPromotedValue(BinOp->getOperand(0), NonStdType, PromotedTy, Builder, Indent, PromotedValues);
  Value *RHS = getPromotedValue(BinOp->getOperand(1), NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // Determine the target type for the new binary operation
  Type* TargetType = NeedsPromotion ? PromotedTy : BinOp->getType();

  // Adjust operands to the target type *before* creating the new BinOp
  LHS = adjustType(LHS, TargetType, Builder, Indent + "  Adjusting LHS: ");
  RHS = adjustType(RHS, TargetType, Builder, Indent + "  Adjusting RHS: ");

  // Now LHS and RHS must have the same type
  assert(LHS->getType() == RHS->getType() && "Operand types mismatch for BinOp after adjustment!");
  Value *NewInst = Builder.CreateBinOp(BinOp->getOpcode(), LHS, RHS);

  LLVM_DEBUG(dbgs() << Indent << "  " << *BinOp << "   promoting BinOp: ====> "
                    << *NewInst << "\n");
  PromotedValues[BinOp] = NewInst;
  Replacements.push_back(Replacement(BinOp, NewInst));
}

static void processSelectInst(SelectInst *SelI, Type *NonStdType, Type *PromotedTy,
                              IRBuilder<> &Builder, const std::string &Indent,
                              SmallVectorImpl<Replacement> &Replacements,
                              SmallDenseMap<Value *, Value *> &PromotedValues) {
  bool NeedsPromotion = isNonStandardInt(SelI->getType());

  // Get potentially promoted operands
  Value *Condition = SelI->getCondition(); // Condition is usually i1
  Value *TrueVal = getPromotedValue(SelI->getTrueValue(), NonStdType, PromotedTy, Builder, Indent, PromotedValues);
  Value *FalseVal = getPromotedValue(SelI->getFalseValue(), NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // Make sure condition is i1
  if (Condition->getType() != Type::getInt1Ty(SelI->getContext())) {
    LLVM_DEBUG(dbgs() << Indent << "    Converting condition to i1: " << *Condition << "\n");
    // Use adjustType to handle potential promotion of condition operand if needed
    Condition = adjustType(Condition, Type::getInt1Ty(SelI->getContext()), Builder, Indent + "  Adjusting Cond: ");
    // Check if adjustment resulted in non-i1; if so, create comparison
    if (Condition->getType() != Type::getInt1Ty(SelI->getContext())) {
         Condition = Builder.CreateICmpNE(
            Condition,
            Constant::getNullValue(Condition->getType()),
            "select.cond");
    }
  }

  // Determine the target type for the select result and true/false values
  Type* TargetType = NeedsPromotion ? PromotedTy : SelI->getType();

  // Adjust TrueVal and FalseVal to the target type *before* creating the new SelectInst
  TrueVal = adjustType(TrueVal, TargetType, Builder, Indent + "  Adjusting TrueVal: ");
  FalseVal = adjustType(FalseVal, TargetType, Builder, Indent + "  Adjusting FalseVal: ");

  // Now TrueVal and FalseVal must have the same type
  assert(TrueVal->getType() == FalseVal->getType() && "Operand types mismatch for Select after adjustment!");
  Value *NewSelect = Builder.CreateSelect(Condition, TrueVal, FalseVal, SelI->getName());

  LLVM_DEBUG(dbgs() << Indent << "  " << *SelI << "   promoting Select: ====> "
                    << *NewSelect << "\n");
  PromotedValues[SelI] = NewSelect;
  Replacements.push_back(Replacement(SelI, NewSelect));
}

static void processICmpInst(ICmpInst *CmpI, Type *NonStdType, Type *PromotedTy,
                            IRBuilder<> &Builder, const std::string &Indent,
                            SmallVectorImpl<Replacement> &Replacements,
                            SmallDenseMap<Value *, Value *> &PromotedValues) {
  // Get potentially promoted operands
  Value *LHS = getPromotedValue(CmpI->getOperand(0), NonStdType, PromotedTy, Builder, Indent, PromotedValues);
  Value *RHS = getPromotedValue(CmpI->getOperand(1), NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // Determine the common type for comparison. Prefer the wider type if they differ.
  // If either became PromotedTy, use that.
  Type *CompareType = nullptr;
  Type* LHSType = LHS->getType();
  Type* RHSType = RHS->getType();

  bool LHSIsPromoted = LHSType == PromotedTy || (isa<IntegerType>(LHSType) && HipPromoteIntsPass::getPromotedBitWidth(LHSType->getIntegerBitWidth()) == PromotedTy->getIntegerBitWidth());
  bool RHSIsPromoted = RHSType == PromotedTy || (isa<IntegerType>(RHSType) && HipPromoteIntsPass::getPromotedBitWidth(RHSType->getIntegerBitWidth()) == PromotedTy->getIntegerBitWidth());

  if (LHSIsPromoted || RHSIsPromoted) {
    CompareType = PromotedTy; // Promote comparison to the wider type if either operand involves it
  } else if (LHSType->getPrimitiveSizeInBits() > RHSType->getPrimitiveSizeInBits()) {
    CompareType = LHSType;
  } else {
    CompareType = RHSType; // Use RHS type if it's wider or if types are the same
  }
   LLVM_DEBUG(dbgs() << Indent << "    Determined ICmp CompareType: " << *CompareType << "\n");

  // Adjust operands *to the determined comparison type*
  LHS = adjustType(LHS, CompareType, Builder, Indent + "  Adjusting ICmp LHS: ");
  RHS = adjustType(RHS, CompareType, Builder, Indent + "  Adjusting ICmp RHS: ");

  // Now LHS and RHS must have the same type
  assert(LHS->getType() == RHS->getType() && "Operand types mismatch for ICmp after adjustment!");
  // Create new comparison instruction (result is always i1)
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
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        bool NeedsPromotion = false;
        // Check result type
        if (auto *IntTy = dyn_cast<IntegerType>(I.getType())) {
          if (!isStandardBitWidth(IntTy->getBitWidth())) {
            NeedsPromotion = true;
          }
        }
        // Check operand types
        if (!NeedsPromotion) {
          for (Value *Op : I.operands()) {
            if (auto *OpIntTy = dyn_cast<IntegerType>(Op->getType())) {
              if (!isStandardBitWidth(OpIntTy->getBitWidth())) {
                NeedsPromotion = true;
                break; // Found a non-standard operand, no need to check others
              }
            }
          }
        }

        if (NeedsPromotion) {
            // Check if it's already in the worklist to avoid duplicates
            bool Found = false;
            for(Instruction *ExistingI : WorkList) {
                if (ExistingI == &I) {
                    Found = true;
                    break;
                }
            }
            if (!Found) {
                 LLVM_DEBUG(dbgs() << "Adding instruction to worklist due to non-standard type: " << I << "\n");
                 WorkList.push_back(&I);
            }
        }
      }
    }

    // Process the worklist
    for (Instruction *I : WorkList) {
      // Skip if we've already processed this instruction as part of another chain
      if (GlobalVisited.count(I))
        continue;

      // Determine the non-standard type and the target promoted type that trigger processing for this instruction
      Type* NonStdTriggerType = nullptr;
      Type* PromotedTargetType = nullptr;

      // Check result type first
      if (auto *ResIntTy = dyn_cast<IntegerType>(I->getType())) {
        if (!isStandardBitWidth(ResIntTy->getBitWidth())) {
          NonStdTriggerType = ResIntTy;
          PromotedTargetType = Type::getIntNTy(M.getContext(), HipPromoteIntsPass::getPromotedBitWidth(ResIntTy->getBitWidth()));
           LLVM_DEBUG(dbgs() << "Triggering promotion for " << *I << " due to non-standard result type: " << *NonStdTriggerType << "\n");
        }
      }

      // If result type is standard, check operands
      if (!NonStdTriggerType) {
        for (Value *Op : I->operands()) {
          if (auto *OpIntTy = dyn_cast<IntegerType>(Op->getType())) {
            if (!isStandardBitWidth(OpIntTy->getBitWidth())) {
              NonStdTriggerType = OpIntTy;
              PromotedTargetType = Type::getIntNTy(M.getContext(), HipPromoteIntsPass::getPromotedBitWidth(OpIntTy->getBitWidth()));
              LLVM_DEBUG(dbgs() << "Triggering promotion for " << *I << " due to non-standard operand type: " << *NonStdTriggerType << " (" << *Op << ")\n");
              break; // Found the first non-standard operand, use it as the trigger
            }
          }
        }
      }

      // If we found a non-standard type that requires promotion
      if (NonStdTriggerType && PromotedTargetType) {
          SmallVector<Replacement, 16> Replacements;
          SmallDenseMap<Value *, Value *> PromotedValues;
          SmallVector<PendingPhiAdd, 16> PendingPhiAdds;

          // Call promoteChain using the identified non-standard type and its corresponding promoted type
          // GlobalVisited is passed to track processed instructions across different chain roots
          promoteChain(I, NonStdTriggerType, PromotedTargetType, GlobalVisited, Replacements,
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
                LLVM_DEBUG(dbgs() << "    Adjusting type for PHI add from " << *PromotedValue->getType() << " to " << *TargetPhi->getType() << "\n");
                PromotedValue = adjustType(PromotedValue, TargetPhi->getType(), PhiBuilder, "      "); // Use helper
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
                // Replace uses only if the user instruction itself is part of the processed chain
                // or if it's not an instruction (e.g. a constant expression using the old value).
                Instruction* UserInst = dyn_cast<Instruction>(U);
                if (!UserInst || GlobalVisited.count(UserInst)) {
                     LLVM_DEBUG(dbgs() << "  Updating use in: " << *U << "\n");
                     U->replaceUsesOfWith(R.Old, R.New);
                } else {
                     LLVM_DEBUG(dbgs() << "  Skipping update for use in unprocessed instruction: " << *U << "\n");
                }
            }
          }

          // Then, for any instructions with remaining uses, force replacement.
          // This can happen with complex dependencies or cycles not fully resolved by the above.
          for (auto &R : Replacements) {
            if (!R.Old->use_empty()) {
              LLVM_DEBUG(dbgs() << "Instruction still has uses after initial replacement: " << *R.Old << "\n" << "  Forcing replaceAllUsesWith...\n");
              R.Old->replaceAllUsesWith(R.New);
            }
          }

          // Finally, delete the original instructions in reverse order to handle dependencies
          for (auto It = Replacements.rbegin(); It != Replacements.rend(); ++It) {
            LLVM_DEBUG(dbgs() << "Deleting instruction: " << *(It->Old) << "\n");
            if (!It->Old->use_empty()) {
              LLVM_DEBUG(dbgs() << "WARNING: Instruction still has uses before deletion: " << *(It->Old) << "\n");
              // Force replacement again just in case
              It->Old->replaceAllUsesWith(It->New);
            }
            It->Old->eraseFromParent();
          }

          Changed = true; // Mark that changes were made
      } // End if (NonStdTriggerType && PromotedTargetType)
    } // End for (Instruction *I : WorkList)
  } // End for (Function &F : M)

  // Print the final IR state before exiting
  LLVM_DEBUG(dbgs() << "Final module IR after HipPromoteIntsPass:\n");
  LLVM_DEBUG(M.print(dbgs(), nullptr));

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}