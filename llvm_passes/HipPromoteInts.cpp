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
 *
 * === Promotion Walkthrough with PHI ===
 * Original IR (Illustrating PHI with standard->non-standard trunc):
 *
 *   ; Assume %cond (i1), %std_val (i64), %non_std_val (i33) are defined earlier.
 *   entry:
 *     br i1 %cond, label %bb.true, label %bb.false
 *
 *   bb.true: ; Original Path
 *     ; --- Originals still exist ---
 *     %val_true = trunc i64 %std_val to i33
 *     br label %bb.merge
 *
 *   bb.false: ; Original Path
 *     %val_false = add i33 %non_std_val, 1
 *     br label %bb.merge
 *
 *   bb.merge: ; Original Path
 *     %merged_val = phi i33 [ %val_true, %bb.true ], [ %val_false, %bb.false ]
 *     %final_res = trunc i33 %merged_val to i8
 *     ; Use %final_res (i8)
 *
 *   ; --- New Promoted Instructions/Values --- (inserted somewhere, possibly different BBs)
 *     %non_std_val.promoted = zext i33 %non_std_val to i64 ; (Created by getPromotedValue)
 *     %const1_promoted = i64 1                               ; (Created by getPromotedValue)
 *     ; --- Promoted path for bb.false ---
 *     %val_false.promoted = add i64 %non_std_val.promoted, %const1_promoted
 *     ; --- Promoted PHI ---
 *     %merged_val.promoted = phi i64 [ %std_val, %bb.true ], [ %val_false.promoted, %bb.false ]
 *     ; --- Promoted final truncation ---
 *     %final_res.promoted = trunc i64 %merged_val.promoted to i8
 *
 * 1. Initial State (Non-Standard Type i33):
 *   The pass starts with the original IR as shown just above.
 *
 * 2. Intermediate State (During Promotion - Before Replacements):
 *   - The pass identifies instructions involving i33: `%val_true`, `%val_false`, `%merged_val`, `%final_res`.
 *   - It starts processing, creating promoted counterparts.
 *   - When processing `%merged_val` (a PHINode):
 *     - A new PHI `%merged_val.promoted` of type i64 is created.
 *     - For incoming `%val_true` (from `trunc i64 %std_val to i33`):
 *       - The special logic in `processPhiNode` recognizes this standard->non-standard trunc.
 *       - It uses the *source* of the trunc, `%std_val` (already i64), directly as the incoming value.
 *     - For incoming `%val_false` (from `add i33 %non_std_val, 1`):
 *       - `getPromotedValue` is called for `%val_false`.
 *       - This recursively processes `%val_false`, creating `%val_false.promoted = add i64 %non_std_val.promoted, 1`.
 *       - `%val_false.promoted` (i64) is used as the incoming value.
 *     - An intermediate `%final_res.promoted = trunc i64 %merged_val.promoted to i8` is created.
 *
 * 3. Final State (After Replacements, Deletions, and Cleanup):
 *   - Uses of original instructions (`%val_true`, `%val_false`, `%merged_val`, `%final_res`)
 *     are replaced with their final values (`%std_val`, `%val_false.promoted`, `%merged_val.promoted`, `%final_res.promoted`).
 *   - Original instructions are deleted.
 *   - Any intermediate bridging instructions (like potential zexts created by getPromotedValue)
 *     become dead code and are removed by cleanup passes.
 *   - Cleanup passes (like ADCE) remove the dead code.
 *   ; --- Final Snippet after Promotion and Cleanup ---
 *   entry:
 *     br i1 %cond, label %bb.true.final, label %bb.false.final
 *
 *   bb.true.final:
 *     ; Path becomes empty as the original trunc was bypassed and deleted.
 *     br label %bb.merge.final
 *
 *   bb.false.final:
 *     %const1_promoted = i64 1
 *     %val_false.promoted = add i64 %non_std_val.promoted, %const1_promoted
 *     br label %bb.merge.final
 *
 *   bb.merge.final:
 *     %merged_val.promoted = phi i64 [ %std_val, %bb.true.final ], [ %val_false.promoted, %bb.false.final ]
 *     %final_res.promoted = trunc i64 %merged_val.promoted to i8
 *     ; Use %final_res.promoted (i8)
 *
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

// Helper to check for non-standard integer types
static bool isNonStandardInt(Type *T) {
  if (auto *IntTy = dyn_cast<IntegerType>(T)) {
    return !HipPromoteIntsPass::isStandardBitWidth(IntTy->getBitWidth());
  }
  return false;
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

// Helper to get or create promoted value
// This helper function is called whenever an instruction processing function
// (e.g., `processBinaryOperator`, `processPhiNode`) needs an operand's value.
// It ensures that if the operand `V` is part of the non-standard type chain,
// an equivalent value with the `PromotedTy` is returned. It handles:
// 1. Returning cached results from `PromotedValues`.
// 2. Returning `V` directly if it's already the `PromotedTy`.
// 3. Promoting `ConstantInt`s of `NonStdType` directly to `PromotedTy`.
// 4. Creating `ZExt`/`Trunc`/`BitCast` instructions if `V` is an instruction
//    result of `NonStdType` to convert it to `PromotedTy`.
// 5. Returning other values (standard constants, arguments) directly.
//
// Parameters:
//   V: The original Value* operand.
//   NonStdType: The specific non-standard type being processed in the current chain (e.g., i33).
//   PromotedTy: The target standard type for the current chain (e.g., i64).
//   Builder: IRBuilder positioned correctly to insert new instructions if needed.
//   Indent: String for debug printing indentation.
//   PromotedValues: Map to cache already promoted values.
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
  if (V->getType() == NonStdType) {
    // Sanity check if it's an instruction that should have been processed already
    assert(isa<Instruction>(V) && !PromotedValues.count(V) && "Encountered unprocessed non-standard instruction");

    // Use adjustType to handle the conversion
    Value *NewV = adjustType(V, PromotedTy, Builder, Indent + "      ");

    PromotedValues[V] = NewV;
    return NewV;
  }

  // Otherwise return original value
  LLVM_DEBUG(dbgs() << Indent << "      Using original value: " << *V << "\n");
  return V;
};

static inline void finalizePromotion(Instruction *Old, Value *New, 
                                     SmallVectorImpl<Replacement> &Replacements,
                                     SmallDenseMap<Value *, Value *> &PromotedValues) {
    PromotedValues[Old] = New; 
    Replacements.push_back(Replacement(Old, New));
}

// Helper to handle TruncInst as incoming PHI value
static bool handlePhiIncomingTrunc(TruncInst *TruncI, Type *NonStdType,
                                   Type *PromotedType, const std::string &Indent,
                                   SmallDenseMap<Value *, Value *> &PromotedValues,
                                   SmallVectorImpl<PendingPhiAdd> &PendingPhiAdds,
                                   PHINode *NewPhi, BasicBlock *IncomingBlock,
                                   Value *&NewIncomingValue) {
  Value *TruncSrc = TruncI->getOperand(0);
  Type *SrcTy = TruncSrc->getType();
  Type *DestTy = TruncI->getDestTy();

  bool IsSrcStandard = HipPromoteIntsPass::isStandardBitWidth(
      SrcTy->isIntegerTy() ? SrcTy->getIntegerBitWidth() : 0);
  bool IsDestNonStandard = !HipPromoteIntsPass::isStandardBitWidth(
      DestTy->isIntegerTy() ? DestTy->getIntegerBitWidth() : 0);

  // Case 1: Standard -> Non-Standard Truncation
  if (IsSrcStandard && IsDestNonStandard) {
    LLVM_DEBUG(dbgs() << Indent
                      << "      Found standard -> non-standard trunc: "
                      << *TruncI << "\n");
    if (PromotedValues.count(TruncI)) {
      NewIncomingValue = PromotedValues[TruncI];
      LLVM_DEBUG(dbgs() << Indent << "      Using existing promoted value: "
                        << *NewIncomingValue << "\n");
    } else {
      Value *PromotedSrc =
          PromotedValues.count(TruncSrc) ? PromotedValues[TruncSrc] : TruncSrc;
      LLVM_DEBUG(dbgs() << Indent
                        << "      Using source directly: " << *PromotedSrc
                        << "\n");
      NewIncomingValue = PromotedSrc;
      PromotedValues[TruncI] = NewIncomingValue; // Cache the bypass
    }
    return false; // Not deferred
  }
  // Case 2: Non-Standard Source Truncation
  else if (!IsSrcStandard) {
    LLVM_DEBUG(dbgs() << Indent
                      << "      Found non-standard source trunc: " << *TruncI
                      << "\n");
    if (PromotedValues.count(TruncI)) {
      NewIncomingValue = PromotedValues[TruncI];
      LLVM_DEBUG(dbgs() << Indent
                        << "      Using existing promoted value: " << *NewIncomingValue
                        << "\n");
      return false; // Not deferred
    } else {
      LLVM_DEBUG(dbgs() << Indent
                        << "      Deferring PHI add for unprocessed non-standard "
                           "truncation\n");
      PendingPhiAdds.push_back({NewPhi, TruncI, IncomingBlock});
      return true; // Deferred
    }
  }
  // Case 3: Other Truncation (e.g., Standard -> Standard)
  return false; // Not handled specially, let general logic take over
}

// Helper to handle ZExtInst as incoming PHI value
static bool handlePhiIncomingZExt(ZExtInst *ZExtI, Type *NonStdType,
                                  Type *PromotedType, const std::string &Indent,
                                  SmallDenseMap<Value *, Value *> &PromotedValues,
                                  SmallVectorImpl<PendingPhiAdd> &PendingPhiAdds,
                                  PHINode *NewPhi, BasicBlock *IncomingBlock,
                                  Value *&NewIncomingValue) {
  Value *ZExtSrc = ZExtI->getOperand(0);
  if (auto *SrcTy = dyn_cast<IntegerType>(ZExtSrc->getType())) {
    if (!HipPromoteIntsPass::isStandardBitWidth(SrcTy->getBitWidth())) {
      LLVM_DEBUG(dbgs() << Indent
                        << "      Found zero-extension from non-standard type: "
                        << *ZExtI << "\n");
      if (PromotedValues.count(ZExtI)) {
        NewIncomingValue = PromotedValues[ZExtI];
        LLVM_DEBUG(dbgs() << Indent << "      Using existing promoted value for zext: "
                          << *NewIncomingValue << "\n");
        return false; // Not deferred
      } else {
        LLVM_DEBUG(
            dbgs() << Indent
                   << "      Deferring PHI add for unprocessed non-standard zext\n");
        PendingPhiAdds.push_back({NewPhi, ZExtI, IncomingBlock});
        return true; // Deferred
      }
    }
  }
  return false; // Not a non-standard ZExt, let general logic take over
}

// Helper to get/adjust the incoming value for the new PHI node
static bool
determinePhiIncomingValue(Value *IncomingValue, Type *NonStdType,
                          Type *PromotedType, PHINode *Phi,
                          const std::string &Indent,
                          SmallDenseMap<Value *, Value *> &PromotedValues,
                          SmallVectorImpl<PendingPhiAdd> &PendingPhiAdds,
                          PHINode *NewPhi, BasicBlock *IncomingBlock,
                          Value *&NewIncomingValue) {
  // Check cache first
  if (PromotedValues.count(IncomingValue)) {
    NewIncomingValue = PromotedValues[IncomingValue];
    LLVM_DEBUG(dbgs() << Indent << "      Using existing promoted value: "
                      << *NewIncomingValue << "\n");
    return false; // Not deferred
  }

  // Handle Instruction types
  if (auto *IncomingInst = dyn_cast<Instruction>(IncomingValue)) {
    // Check if the instruction itself or its operands involve non-std type
    if (isNonStandardInt(IncomingInst->getType()) ||
        llvm::any_of(IncomingInst->operands(),
                     [](Value *V) { return isNonStandardInt(V->getType()); })) {
      // Defer if it involves non-standard types and hasn't been processed
      LLVM_DEBUG(dbgs() << Indent
                        << "      Deferring PHI add for unprocessed instruction "
                           "involving non-std type: "
                        << *IncomingValue << "\n");
      PendingPhiAdds.push_back({NewPhi, IncomingValue, IncomingBlock});
      return true; // Deferred
    } else {
      // Standard instruction, get its promoted value (might be itself)
      IRBuilder<> TmpBuilder(Phi); // Builder before the PHI
      NewIncomingValue = getPromotedValue(IncomingValue, NonStdType, PromotedType,
                                          TmpBuilder, Indent + "      ",
                                          PromotedValues);
      LLVM_DEBUG(dbgs() << Indent
                        << "      Using potentially promoted value for standard "
                           "instruction: "
                        << *NewIncomingValue << "\n");
      return false; // Not deferred
    }
  } else {
    // Must be a constant, argument, global - get its potentially promoted form
    IRBuilder<> TmpBuilder(Phi); // Builder before the PHI
    NewIncomingValue = getPromotedValue(IncomingValue, NonStdType, PromotedType,
                                        TmpBuilder, Indent + "      ",
                                        PromotedValues);
    LLVM_DEBUG(dbgs() << Indent << "      Using potentially promoted value: "
                      << *NewIncomingValue << "\n");
    return false; // Not deferred
  }
}

static void processPhiNode(PHINode *Phi, Type *NonStdType, Type *PromotedTy,
                           IRBuilder<> &Builder, const std::string &Indent,
                           SmallVectorImpl<Replacement> &Replacements,
                           SmallDenseMap<Value *, Value *> &PromotedValues,
                           SmallVectorImpl<PendingPhiAdd> &PendingPhiAdds) {
  // Create new PHI node with the promoted type
  Type *PromotedType = HipPromoteIntsPass::getPromotedType(Phi->getType());
  PHINode *NewPhi =
      PHINode::Create(PromotedType, Phi->getNumIncomingValues(), "", Phi);

  // Register the PHI node early to handle cycles
  LLVM_DEBUG(dbgs() << Indent << "  Creating promotion for PHI: " << *Phi
                    << " to " << *NewPhi << "\n");
  PromotedValues[Phi] = NewPhi;

  // Temporary list for values to add immediately
  SmallVector<std::pair<Value *, BasicBlock *>, 4> ValuesToAdd;

  // Iterate through original incoming values
  for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
    Value *IncomingValue = Phi->getIncomingValue(i);
    BasicBlock *IncomingBlock = Phi->getIncomingBlock(i);
    LLVM_DEBUG(dbgs() << Indent << "    Processing incoming value: " << *IncomingValue
                      << " from block: " << IncomingBlock->getName() << "\n");

    Value *NewIncomingValue = nullptr;
    bool DeferAdd = false;

    // Check for special cases first (Trunc, ZExt)
    if (auto *TruncI = dyn_cast<TruncInst>(IncomingValue)) {
      DeferAdd = handlePhiIncomingTrunc(TruncI, NonStdType, PromotedType, Indent,
                                        PromotedValues, PendingPhiAdds, NewPhi,
                                        IncomingBlock, NewIncomingValue);
    } else if (auto *ZExtI = dyn_cast<ZExtInst>(IncomingValue)) {
      DeferAdd = handlePhiIncomingZExt(ZExtI, NonStdType, PromotedType, Indent,
                                       PromotedValues, PendingPhiAdds, NewPhi,
                                       IncomingBlock, NewIncomingValue);
    }

    // If not handled by special cases and not already deferred, use general logic
    if (!NewIncomingValue && !DeferAdd) {
      DeferAdd = determinePhiIncomingValue(
          IncomingValue, NonStdType, PromotedType, Phi, Indent, PromotedValues,
          PendingPhiAdds, NewPhi, IncomingBlock, NewIncomingValue);
    }

    // If we determined a value and didn't defer, add it to the list
    if (!DeferAdd) {
      assert(NewIncomingValue && "Value for PHI should have been determined");
      if (NewIncomingValue->getType() != PromotedType) {
        IRBuilder<> PhiBuilder(Phi); // Builder before the PHI
        LLVM_DEBUG(dbgs() << Indent
                          << "      Adjusting incoming PHI value type: "
                          << *NewIncomingValue << " to " << *PromotedType
                          << "\n");
        NewIncomingValue = adjustType(NewIncomingValue, PromotedType, PhiBuilder,
                                      Indent + "        Adjusting PHI incoming: ");
      }
      ValuesToAdd.push_back({NewIncomingValue, IncomingBlock});
      LLVM_DEBUG(dbgs() << Indent << "      Queueing PHI add: ["
                        << *NewIncomingValue << ", " << IncomingBlock->getName()
                        << "]\n");
    }
  } // End loop through original incoming values

  // Add the non-deferred values collected
  LLVM_DEBUG(dbgs() << Indent << "  Adding " << ValuesToAdd.size()
                    << " non-deferred incoming values to PHI: " << *NewPhi
                    << "\n");
  for (const auto &Pair : ValuesToAdd) {
    NewPhi->addIncoming(Pair.first, Pair.second);
  }

  // Calculate pending count separately for clarity in debug message
  size_t PendingCount = 0;
  for(const auto& p : PendingPhiAdds) {
      if (p.TargetPhi == NewPhi) {
          PendingCount++;
      }
  }

  LLVM_DEBUG(dbgs() << Indent << "  " << *Phi
                    << "   promoting PHI node: ====> " << *NewPhi << " ("
                    << NewPhi->getNumIncomingValues() << " initial incoming, "
                    << PendingCount << " pending)\n");

  Replacements.push_back(Replacement(Phi, NewPhi));
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
    finalizePromotion(ZExtI, NewValue, Replacements, PromotedValues);

  } else if (IsDestNonStandard) {
    // Case 2: Destination is Non-Standard (e.g., zext i32 -> i56)
    Type *PromotedDestTy = HipPromoteIntsPass::getPromotedType(DestTy); // e.g., i64

    // Since the source is standard, getPromotedValue should have returned a value
    // with the original source type SrcTy. No adjustment should be needed.
    assert(PromotedSrc->getType() == SrcTy && "Promoted source type mismatch for standard ZExt source");

    // Create ZExt using the standard source (PromotedSrc) to the *promoted* destination type.
    NewValue = Builder.CreateZExt(PromotedSrc, PromotedDestTy);
    LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (non-std dest): ====> " << *NewValue << "\n");
    finalizePromotion(ZExtI, NewValue, Replacements, PromotedValues);

  } else {
    // Case 3: Source and Destination are Standard (e.g., zext i32 -> i64)
    // Recreate the instruction. Since the source is standard, getPromotedValue
    // should have returned a value with the original source type SrcTy.
    assert(PromotedSrc->getType() == SrcTy && "Promoted source type mismatch for standard ZExt source");

    // Create the ZExt with the standard source (PromotedSrc) and original standard destination type.
    NewValue = Builder.CreateZExt(PromotedSrc, DestTy);
    LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (standard src/dest): ====> " << *NewValue << "\n");
    // Map original ZExt result to new ZExt result. Both should have same standard type.
    finalizePromotion(ZExtI, NewValue, Replacements, PromotedValues);
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
    // Use adjustType to ensure the source matches the promoted type
    PromotedSrc = adjustType(PromotedSrc, PromotedTy, Builder, Indent + "    Adjusting source to promoted type: ");
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

  // Create a new trunc for external users using adjustType
  Value *NewTrunc = adjustType(PromotedSrc, DestTy, Builder, Indent + "  Creating external trunc: ");
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
  finalizePromotion(BinOp, NewInst, Replacements, PromotedValues);
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
  finalizePromotion(SelI, NewSelect, Replacements, PromotedValues);
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
  finalizePromotion(CmpI, NewCmp, Replacements, PromotedValues);
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
      // Use adjustType to convert argument back to expected type
      LLVM_DEBUG(dbgs() << Indent << "    Adjusting argument " << i << " from " << *NewArg->getType() << " to " << *ExpectedType << "\n");
      NewArg = adjustType(NewArg, ExpectedType, Builder, Indent + "      Adjusting arg: ");
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
  finalizePromotion(OldCall, NewCall, Replacements, PromotedValues);
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
    // Use adjustType to convert the stored value back to the expected type
    LLVM_DEBUG(dbgs() << Indent << "    Adjusting store value from " << *NewStoredValue->getType() << " to " << *ExpectedType << "\n");
    NewStoredValue = adjustType(NewStoredValue, ExpectedType, Builder, Indent + "      Adjusting store val: ");
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

      // Use adjustType to promote the loaded value
      LLVM_DEBUG(dbgs() << Indent << "    Promoting loaded non-standard value using adjustType\n");
      PromotedValue = adjustType(NewLoad, PromotedLoadType, AfterLoadBuilder, Indent + "      Adjusting load result: ");

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
      // Use adjustType to match the function's return type
      LLVM_DEBUG(dbgs() << Indent << "    Adjusting return value to match function type\n");
      NewRetVal = adjustType(NewRetVal, FuncRetType, Builder, Indent + "      Adjusting ret val: ");
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