#include "HipPromoteInts.h"
#include "llvm/ADT/SmallVector.h" // Include for SmallVector
#include "llvm/ADT/SmallPtrSet.h" // Include for SmallPtrSet
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <functional>

#define DEBUG_TYPE "hip-promote-ints"

/**
 * @brief Promotes non-standard integer types (e.g., i33, i56) to the next standard width.
 *
 * LLVM's loop optimizations can generate integer types with bit widths that are
 * not powers of two (or 1). These non-standard types can cause issues during
 * later stages, particularly SPIR-V translation.
 * Key Data Structures:
 * -------------------- 
 * - `WorkList`: Stores initial instructions needing promotion within a function.
 * - `GlobalVisited`: Tracks instructions already processed across *all* chains in a function to avoid redundant work.
 *                    Ensures each instruction is promoted only once per function, even if reachable
 *                    from multiple starting points in the `WorkList`.
 * - `PromotedValues`: Maps original `Value*` (instructions, constants) to their promoted `Value*` equivalents
 *   within the context of a single `promoteChain` call.
 * - `Replacements`: Stores pairs of `{original instruction, new value}` created during `promoteChain`.
 * - `PendingPhiAdds`: Temporarily stores `{TargetPhi, OriginalValue, IncomingBlock}` tuples for PHI inputs
 *   whose `OriginalValue` hadn't been processed yet when `processPhiNode` was called.
 * Algorithm Overview:
 * ------------------
 * This pass replaces instructions involving non-standard integer types with
 * equivalent instructions using the next largest standard integer type (8, 16, 32, 64).
 * It operates in two main phases within the `run` method for each function:
 *
 * 1. Worklist Collection:
 *    - Identify all instructions that either produce a non-standard integer type
 *      or use an operand with a non-standard integer type.
 *    - These instructions form the initial `WorkList` for promotion chains.
 *
 * 2. Chain Promotion and Replacement:
 *    - For each instruction in the `WorkList` (that hasn't already been processed
 *      via `GlobalVisited`):
 *        a. **Recursive Promotion (`promoteChain`)**: Recursively traverse the use-def chain
 *           starting from the initial instruction. For each instruction encountered:
 *           - Dispatch to the appropriate `processInstruction` handler (e.g.,
 *             `processBinaryOperator`, `processPhiNode`, `processTruncInst`).
 *           - Determine its corresponding promoted instruction/value (using standard types).
 *           - Store the mapping from the original instruction/value to its promoted counterpart
 *             in the `PromotedValues` map for the current chain.
 *           - Add the original instruction and its replacement/new value to a `Replacements` list.
 *           - Use the `adjustType` helper function to insert explicit type conversions
 *             (zext/trunc/bitcast) as needed to maintain type consistency *within the newly built
 *             promoted instruction chain*.
 *           - Special handling for PHI nodes (`processPhiNode`): Create the new PHI node
 *             with the promoted type and register it immediately in `PromotedValues` to break
 *             potential cycles during recursive processing. Defer adding incoming values
 *             if their producers haven't been processed yet by adding them to `PendingPhiAdds`.
 *           - Special handling for `TruncInst` and `ZExtInst` to manage conversions between
 *             standard and non-standard types, sometimes bypassing the instruction or using
 *             the promoted source directly.
 *        b. **PHI Node Patching**: After processing a chain initiated by `promoteChain`,
 *           iterate through the `PendingPhiAdds` list generated during that chain's processing.
 *           Look up the now-available promoted values for the original incoming values (using
 *           `PromotedValues`), adjust their types if necessary using `adjustType`, and add
 *           them to their corresponding new PHI nodes.
 *        c. **Instruction Replacement**: Iterate through the `Replacements` list.
 *           - First pass: Replace uses of the original instruction `Old` with the new value `New`,
 *             *only* updating users that were also part of the processed chain (tracked in `GlobalVisited`).
 *           - Second pass: Perform a final `replaceAllUsesWith` on `Old` to catch any remaining uses
 *             (e.g., users outside the processed chain or complex dependencies).
 *           - Erase the original instruction `Old` (in reverse order of replacement). 
 *
 * Example: Promotion Walkthrough with PHI Deferral
 * --------
 * Original IR (Illustrating standard->non-standard trunc and cycle):
 *
 *   entry:
 *     %std_val = ... ; i64
 *     %non_std_init = ... ; i33
 *     br label %loop_header
 *
 *   loop_header:
 *     %phi.loop = phi i33 [ %non_std_init, %entry ], [ %next_val, %loop_latch ]
 *     br label %loop_body
 *
 *   loop_body:
 *     %trunc_std = trunc i64 %std_val to i33 ; Standard -> Non-standard
 *     %added = add i33 %phi.loop, 1
 *     br label %loop_latch
 *
 *   loop_latch:
 *     %next_val = select i1 ..., i33 %added, i33 %trunc_std
 *     br label %loop_header
 *
 * 1. Worklist: Instructions involving `i33` are added: `%phi.loop`, `%added`, `%trunc_std`, `%next_val`.
 * 2. `promoteChain` starts (e.g., triggered by `%added`):
 *    - Processes `%added`: Creates `%added.promoted = add i64 ...`. Needs promoted `%phi.loop`.
 *    - Recursively calls `promoteChain` for `%phi.loop`.
 *    - `processPhiNode` for `%phi.loop`:
 *        - Creates `NewPhi = %phi.loop.promoted = phi i64 ...`.
 *        - Registers `PromotedValues[%phi.loop] = NewPhi`.
 *        - Processes incoming `[%non_std_init, %entry]`: Gets promoted `%non_std_init.promoted`, adds it directly.
 *        - Processes incoming `[%next_val, %loop_latch]`: `PromotedValues` doesn't contain `%next_val` yet.
 *          Adds `{NewPhi, %next_val, %loop_latch}` to `PendingPhiAdds`. Deferral occurs.
 *    - Recursion returns. `promoteChain` continues processing other instructions.
 *    - Processes `%trunc_std`: `processTruncInst` recognizes standard->non-standard. Creates mapping
 *      `PromotedValues[%trunc_std] = %std_val` (bypassing the trunc internally).
 *    - Processes `%next_val`: Creates `%next_val.promoted = select i1 ..., i64 %added.promoted, i64 %std_val`.
 *      Needs promoted `%added` (available) and promoted `%trunc_std` (available as `%std_val`).
 *      Registers `PromotedValues[%next_val] = %next_val.promoted`.
 * 3. PHI Patching: After the chain processing completes:
 *    - Iterates `PendingPhiAdds`. Finds `{NewPhi, %next_val, %loop_latch}`.
 *    - Looks up `PromotedValues[%next_val]`, finds `%next_val.promoted` (i64).
 *    - Calls `NewPhi->addIncoming(%next_val.promoted, %loop_latch)`.
 * 4. Replacement/Deletion: Replaces uses and deletes original `i33` instructions.
 *
 * Final State (Simplified):
 *   entry:
 *     %std_val = ... ; i64
 *     %non_std_init.promoted = zext i33 %non_std_init to i64
 *     br label %loop_header
 *
 *   loop_header:
 *     %phi.loop.promoted = phi i64 [ %non_std_init.promoted, %entry ], [ %next_val.promoted, %loop_latch ]
 *     br label %loop_body
 *
 *   loop_body:
 *     ; %trunc_std deleted
 *     %added.promoted = add i64 %phi.loop.promoted, 1
 *     br label %loop_latch
 *
 *   loop_latch:
 *     %next_val.promoted = select i1 ..., i64 %added.promoted, i64 %std_val
 *     br label %loop_header
 *
 */

using namespace llvm;

// Helper to check for non-standard integer types
static bool isNonStandardInt(Type *T) {
  if (auto *IntTy = dyn_cast<IntegerType>(T)) {
    return !HipPromoteIntsPass::isStandardBitWidth(IntTy->getBitWidth());
  }
  return false;
}

/// @brief Given a linked list of instructions that begin with a non-standard type, 
/// traverse the linked list from the beginning
/// Find the first instruction that either zexts or truncates to the standard type and drop all from there to the end
/// @param LL linked list of instructions
/// @return truncated linked list
static std::vector<Instruction *> truncateUseDefLL(std::vector<Instruction *> LL){
  for (int i = LL.size() - 1; i >= 0; --i) {
    Instruction *Inst = LL[i];
    bool NonStdFound = false;

    // Check if the result type is non-standard
    if (isNonStandardInt(Inst->getType())) {
      NonStdFound = true;
    } else {
      // Check if any operand type is non-standard
      for (Value *Op : Inst->operands()) {
        if (isNonStandardInt(Op->getType())) {
          NonStdFound = true;
          break;
        }
      }
    }

    // If this instruction involves a non-standard type, it's the end of our relevant chain.
    if (NonStdFound) {
      // Return the sublist from the beginning up to and including this instruction.
      return std::vector<Instruction *>(LL.begin(), LL.begin() + i + 1);
    }
  }

  return LL;
}



/// @brief Given an instrucion, return a list of paths in its use-def chain
/// @param I The instruction to get the use-def chain for
/// @return A vector of vectors, where each inner vector represents a distinct path in the use-def chain
static std::vector<std::vector<Instruction *>> getLinkedListsFromUseDefChain(Instruction *I) {
  std::vector<std::vector<Instruction *>> Chains;
  std::set<Instruction *> Visited;
  
  std::function<void(Instruction *, std::vector<Instruction *>)> traverseUsers = 
      [&](Instruction *Inst, std::vector<Instruction *> CurrentChain) {
    if (!Inst || Visited.count(Inst))
      return;
    
    Visited.insert(Inst);
    CurrentChain.push_back(Inst);
    
    // Check if this instruction has any users
    if (Inst->users().empty()) {
      // We've reached the end of a chain, add it to the list of chains
      Chains.push_back(CurrentChain);
      return;
    }
    
    for (User *User : Inst->users()) {
      if (Instruction *UserInst = dyn_cast<Instruction>(User)) {
        // Create a new branch for each user
        traverseUsers(UserInst, CurrentChain);
      }
    }
  };
  
  // Start traversal from the users of I (not including I itself)
  for (User *User : I->users()) {
    if (Instruction *UserInst = dyn_cast<Instruction>(User)) {
      std::vector<Instruction *> NewChain;
      NewChain.push_back(I);
      traverseUsers(UserInst, NewChain);
    }
  }

  // If no chains were found (e.g., instruction has no users), return an empty vector
  if (Chains.empty()) {
    LLVM_DEBUG(dbgs() << "No chains found for: " << *I << "\n");
    return Chains;
  }

  std::vector<std::vector<Instruction *>> truncatedChains;
  // Print the linked lists
  LLVM_DEBUG(dbgs() << "Found " << Chains.size() << " chains for: " << *I << "\n");
  for (unsigned i = 0; i < Chains.size(); ++i) {
    LLVM_DEBUG(dbgs() << "Chain " << i << ":\n");
    for (Instruction *Inst : Chains[i]) {
      LLVM_DEBUG(dbgs() << "  " << *Inst << "\n");
    }
    auto truncatedChain = truncateUseDefLL(Chains[i]);
    truncatedChains.push_back(truncatedChain);
    LLVM_DEBUG(dbgs() << "Truncated chain " << i << ":\n");
    for (Instruction *Inst : truncatedChain) {
      LLVM_DEBUG(dbgs() << "  " << *Inst << "\n");
    }
  }
  
  return truncatedChains;
}



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

// Helper to adjust value V to type TargetTy using Builder
static Value* adjustType(Value *V, Type *TargetTy, IRBuilder<> &Builder, const std::string& Indent = "") {
    if (V->getType() == TargetTy) {
        return V;
    }

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

/**
 * Get or create promoted value
 * 
 * This helper function is called whenever an instruction processing function
 * (e.g., `processBinaryOperator`, `processPhiNode`) needs an operand's value.
 * It ensures that if the operand `V` is part of the non-standard type chain,
 * an equivalent value with the `PromotedTy` is returned. It handles:
 * 1. Returning cached results from `PromotedValues`.
 * 2. Returning `V` directly if it's already the `PromotedTy`.
 * 3. Promoting `ConstantInt`s of `NonStdType` directly to `PromotedTy`.
 * 4. Creating `ZExt`/`Trunc`/`BitCast` instructions if `V` is an instruction
 *    result of `NonStdType` to convert it to `PromotedTy`.
 * 5. Returning other values (standard constants, arguments) directly.
 * 
 * @param V The original Value* operand
 * @param NonStdType The specific non-standard type being processed in the current chain (e.g., i33)
 * @param PromotedTy The target standard type for the current chain (e.g., i64)
 * @param Builder IRBuilder positioned correctly to insert new instructions if needed
 * @param Indent String for debug printing indentation
 * @param PromotedValues Map to cache already promoted values
 * @return The promoted value
 */
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
      LLVM_DEBUG(dbgs() << Indent << "      Returning original NonStd ConstantInt: " << *V << "\n");
      return V; // Return the original ConstantInt
    }
    // If it's a constant of a different standard type, it might need zext/trunc later
    // but we return the original constant for now. Adjustments happen in the instruction processing.
     LLVM_DEBUG(dbgs() << Indent << "      Using original Standard ConstantInt: " << *V << "\n");
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

/**
 * Finalize promotion
 * 
 * This function finalizes the promotion of an instruction.
 * It caches the promoted value and adds the replacement to the list.
 * 
 * @param Old The original instruction
 * @param New The promoted value
 * @param Replacements The list of replacements
 * @param PromotedValues The map of already promoted values
 */
static inline void finalizePromotion(Instruction *Old, Value *New, 
                                     SmallVectorImpl<Replacement> &Replacements,
                                     SmallDenseMap<Value *, Value *> &PromotedValues) {
    PromotedValues[Old] = New; 
    Replacements.push_back(Replacement(Old, New));
}

/**
 * Handle incoming PHI value as TruncInst
 * 
 * This function handles the case where a PHI node has an incoming value that is a TruncInst.
 * It checks if the source of the truncation is standard and the destination is non-standard.
 * If so, it promotes the source value to the promoted type and adds it to the PHI node.
 * 
 * @param TruncI The TruncInst to handle
 * @param NonStdType The non-standard type being promoted
 * @param PromotedType The promoted type
 * @param Indent The indent for debug printing
 * @param PromotedValues The map of already promoted values
 * @param PendingPhiAdds The list of pending PHI node additions
 * @param NewPhi The new PHI node to add
 */
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

/**
 * Handle incoming PHI value as ZExtInst
 * 
 * This function handles the case where a PHI node has an incoming value that is a ZExtInst.
 * It checks if the source of the zero-extension is non-standard.
 * If so, it promotes the source value to the promoted type and adds it to the PHI node.
 * 
 * @param ZExtI The ZExtInst to handle
 * @param NonStdType The non-standard type being promoted
 * @param PromotedType The promoted type
 * @param Indent The indent for debug printing
 * @param PromotedValues The map of already promoted values
 * @param PendingPhiAdds The list of pending PHI node additions
 * @param NewPhi The new PHI node to add
 * @param NewIncomingValue The new incoming value to add
 * @return true if the value was deferred, false otherwise
 */
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

/**
 * Helper to get/adjust the incoming value for the new PHI node
 * 
 * This function checks if the incoming value has already been promoted.
 * If so, it returns the promoted value.
 * Otherwise, it determines the incoming value for the new PHI node.
 * 
 * @param IncomingValue The incoming value to check
 * @param NonStdType The non-standard type being promoted
 * @param PromotedType The promoted type
 * @param Phi The PHI node to add
 * @param Indent The indent for debug printing
 * @param PromotedValues The map of already promoted values
 * @param PendingPhiAdds The list of pending PHI node additions
 * @param NewPhi The new PHI node to add
 * @param IncomingBlock The incoming block to add
 * @param NewIncomingValue The new incoming value to add
 * @return true if the value was deferred, false otherwise
 */
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

  // Get the source value (might be original NonStd ConstantInt)
  Value *PromotedSrc = getPromotedValue(SrcOp, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  bool IsSrcNonStandard = isNonStandardInt(SrcTy);
  bool IsDestNonStandard = isNonStandardInt(DestTy);
  Type *PromotedDestTy = IsDestNonStandard ? HipPromoteIntsPass::getPromotedType(DestTy) : DestTy;

  Value *NewValue = nullptr;

  // Check if SrcOp is a NonStd ConstantInt needing specific extension
  if (isa<ConstantInt>(PromotedSrc) && IsSrcNonStandard) {
      LLVM_DEBUG(dbgs() << Indent << "  Extending NonStd ConstantInt operand for ZExt: " << *PromotedSrc << "\n");
      NewValue = Builder.CreateZExt(PromotedSrc, PromotedDestTy, PromotedSrc->getName() + ".constexpr.zext");
      LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (non-std const src): ====> " << *NewValue << "\n");
      finalizePromotion(ZExtI, NewValue, Replacements, PromotedValues);
      return;
  }

  if (IsSrcNonStandard) {
    // Case 1: Source is Non-Standard (e.g., zext i56 -> i64) (but not a ConstantInt, handled above)
    // PromotedSrc should now have the PromotedTy (e.g., i64)
    assert(PromotedSrc->getType() == PromotedTy && "Non-standard source operand was not promoted correctly in getPromotedValue");

    // Adjust the PromotedSrc (which is i64) to match the promoted destination type (PromotedDestTy)
    NewValue = adjustType(PromotedSrc, PromotedDestTy, Builder, Indent + "    Adjusting NonStd Src for ZExt: ");
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

  Value *OrigLHS = BinOp->getOperand(0);
  Value *OrigRHS = BinOp->getOperand(1);

  // Get potentially promoted operands
  Value *LHS = getPromotedValue(OrigLHS, NonStdType, PromotedTy, Builder, Indent, PromotedValues);
  Value *RHS = getPromotedValue(OrigRHS, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // --- IR Examples after getPromotedValue ---
  // Assume NonStdType = i33, PromotedTy = i64
  //
  // Example 1: %orig_add = add i33 %inst_i33, i33 5
  //   OrigLHS = %inst_i33 (result of some preceding non-std instruction)
  //   OrigRHS = i33 5 (ConstantInt)
  // After getPromotedValue:
  //   LHS = result of adjustType(%inst_i33, i64, ...), likely a zext: '%inst_i33.zext = zext i33 %inst_i33 to i64'
  //   RHS = original ConstantInt* for i33 5 (getPromotedValue no longer promotes constants)
  // Later, extendOrAdjustOperand will see RHS is i33 5 and create a 'zext i33 5 to i64' for the add.
  //
  // Example 2: %orig_sub = sub i64 %std_val, i64 %promoted_non_std
  //   OrigLHS = %std_val (i64)
  //   OrigRHS = %promoted_non_std (i64, already promoted from i33 earlier)
  // After getPromotedValue:
  //   LHS = %std_val (already i64)
  //   RHS = %promoted_non_std (already i64)
  // Later, extendOrAdjustOperand will see types match TargetType (i64) and do nothing.
  //
  // Example 3: %orig_sdiv = sdiv i33 %another_inst, i33 -1
  //   OrigLHS = %another_inst (i33)
  //   OrigRHS = i33 -1 (ConstantInt)
  // After getPromotedValue:
  //   LHS = result of adjustType, e.g., '%another_inst.zext = zext i33 %another_inst to i64'
  //   RHS = original ConstantInt* for i33 -1
  // Later, extendOrAdjustOperand will see RHS is i33 -1 and the opcode is SDiv,
  // so it will create a 'sext i33 -1 to i64' for the sdiv.
  // --- End IR Examples ---

  // Determine the target type for the new binary operation
  Type* TargetType = PromotedTy; // Default to promoted type
  if (!isNonStandardInt(BinOp->getType())) { 
      // If original result was standard, then operands must have been too.
      // Use the original standard type for the new operation.
      TargetType = BinOp->getType();
  }
  LLVM_DEBUG(dbgs() << Indent << "    Determined BinOp TargetType: " << *TargetType << "\n");

  // Extend or adjust operands to the target type *before* creating the new BinOp
  auto extendOrAdjustOperand = 
      [&](Value *Operand, const std::string &Side) -> Value* {
      if (Operand->getType() == TargetType) {
          return Operand; // Already correct type
      }

      // Check if it's a NonStd ConstantInt needing specific extension
      if (isa<ConstantInt>(Operand) && isNonStandardInt(Operand->getType())) {
          LLVM_DEBUG(dbgs() << Indent << "    Extending NonStd ConstantInt " << Side << ": " << *Operand << "\n");
          Instruction::BinaryOps Opcode = BinOp->getOpcode();
          bool NeedsSExt = (Opcode == Instruction::SDiv || Opcode == Instruction::SRem || Opcode == Instruction::AShr);
          
          if (NeedsSExt) {
              LLVM_DEBUG(dbgs() << Indent << "      using SExt to " << *TargetType << " for opcode " << BinOp->getOpcodeName() << "\n");
              return Builder.CreateSExt(Operand, TargetType, Operand->getName() + ".sext");
          } else {
              // Includes: Add, Sub, Mul, UDiv, URem, Shl, LShr, And, Or, Xor
              LLVM_DEBUG(dbgs() << Indent << "      using ZExt to " << *TargetType << " for opcode " << BinOp->getOpcodeName() << "\n");
              return Builder.CreateZExt(Operand, TargetType, Operand->getName() + ".zext");
          }
      } else {
          // Otherwise, use adjustType for general type mismatches
          LLVM_DEBUG(dbgs() << Indent << "    Adjusting " << Side << ": " << *Operand << " to " << *TargetType << "\n");
          return adjustType(Operand, TargetType, Builder, Indent + "      Adjusting BinOp " + Side + ": ");
      }
  };

  LHS = extendOrAdjustOperand(LHS, "LHS");
  RHS = extendOrAdjustOperand(RHS, "RHS");

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
  Value *OrigCond = SelI->getCondition();
  Value *OrigTrue = SelI->getTrueValue();
  Value *OrigFalse = SelI->getFalseValue();

  // Get potentially promoted operands
  Value *Condition = getPromotedValue(OrigCond, NonStdType, PromotedTy, Builder, Indent, PromotedValues); // Condition is usually i1
  Value *TrueVal = getPromotedValue(OrigTrue, NonStdType, PromotedTy, Builder, Indent, PromotedValues);
  Value *FalseVal = getPromotedValue(OrigFalse, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // Make sure condition is i1
  Type *Int1Ty = Type::getInt1Ty(SelI->getContext());
  if (Condition->getType() != Int1Ty) {
    LLVM_DEBUG(dbgs() << Indent << "    Converting condition to i1: " << *Condition << "\n");
    // If condition itself was non-std, it needs promotion first
    if (isNonStandardInt(Condition->getType())) {
        Type* ConditionPromotedTy = Type::getIntNTy(Condition->getContext(), HipPromoteIntsPass::getPromotedBitWidth(Condition->getType()->getIntegerBitWidth()));
        if (isa<ConstantInt>(Condition)) {
             // Assume ZExt for condition constants, unlikely to be signed comparison result
             Condition = Builder.CreateZExt(Condition, ConditionPromotedTy, Condition->getName() + ".zext");
        } else {
            Condition = adjustType(Condition, ConditionPromotedTy, Builder, Indent + "  Adjusting NonStd Cond: ");
        }
    }
    // Create comparison if necessary after potential promotion
    if (Condition->getType() != Int1Ty) {
         Condition = Builder.CreateICmpNE(
            Condition,
            Constant::getNullValue(Condition->getType()),
            SelI->getName() + ".cond"); // Use Select name for clarity
         LLVM_DEBUG(dbgs() << Indent << "      Created NE comparison: " << *Condition << "\n");
    } else {
         LLVM_DEBUG(dbgs() << Indent << "      Condition already i1 after promotion/adjustment." << "\n");
    }
  }

  // Determine the target type for the select result and true/false values
  Type* TargetType = PromotedTy; // Default
   if (!isNonStandardInt(SelI->getType())) { // Original Select result is standard
      // If original result was standard, true/false values must also have been standard.
      // Use the original standard type.
      TargetType = SelI->getType();
  }
  // NOTE: Removed redundant check on operands and isStandardBitWidth call
   LLVM_DEBUG(dbgs() << Indent << "    Determined Select TargetType: " << *TargetType << "\n");

  // Extend or adjust TrueVal and FalseVal to the target type
  auto extendOrAdjustOperand = 
      [&](Value *Operand, const std::string &Side) -> Value* {
      if (Operand->getType() == TargetType) {
          return Operand; // Already correct type
      }
      // Check if it's a NonStd ConstantInt needing specific extension
      if (isa<ConstantInt>(Operand) && isNonStandardInt(Operand->getType())) {
          LLVM_DEBUG(dbgs() << Indent << "    Extending NonStd ConstantInt " << Side << ": " << *Operand << "\n");
          // Select operands usually don't imply signedness, use ZExt
          LLVM_DEBUG(dbgs() << Indent << "      using ZExt to " << *TargetType << "\n");
          return Builder.CreateZExt(Operand, TargetType, Operand->getName() + ".zext");
      } else {
          // Otherwise, use adjustType for general type mismatches
          LLVM_DEBUG(dbgs() << Indent << "    Adjusting " << Side << ": " << *Operand << " to " << *TargetType << "\n");
          return adjustType(Operand, TargetType, Builder, Indent + "      Adjusting Select " + Side + ": ");
      }
  };

  TrueVal = extendOrAdjustOperand(TrueVal, "TrueVal");
  FalseVal = extendOrAdjustOperand(FalseVal, "FalseVal");

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
  Value *OrigLHS = CmpI->getOperand(0);
  Value *OrigRHS = CmpI->getOperand(1);

  // Get potentially promoted operands (getPromotedValue now returns original ConstInts)
  Value *LHS = getPromotedValue(OrigLHS, NonStdType, PromotedTy, Builder, Indent, PromotedValues);
  Value *RHS = getPromotedValue(OrigRHS, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // Determine the common type for comparison. Usually PromotedTy if non-std involved.
  Type *CompareType = PromotedTy; // Default to the wider promoted type
  if (!isNonStandardInt(OrigLHS->getType()) && !isNonStandardInt(OrigRHS->getType())) {
      // Both original operands are standard. Compare using the wider standard type if possible.
      Type* LHSTy = LHS->getType(); // Type after getPromotedValue
      Type* RHSTy = RHS->getType(); // Type after getPromotedValue
      
      // Check if both types are integers before comparing bit widths
      if (LHSTy->isIntegerTy() && RHSTy->isIntegerTy()) {
         unsigned LHSBits = LHSTy->getIntegerBitWidth();
         unsigned RHSBits = RHSTy->getIntegerBitWidth();
         CompareType = LHSBits >= RHSBits ? LHSTy : RHSTy;
      } else {
         assert(false && "Comparisons between non-integer types are not supported");
      }
  }
  // NOTE: Removed potentially unsafe call to getIntegerBitWidth on CompareType here.
   LLVM_DEBUG(dbgs() << Indent << "    Determined ICmp CompareType: " << *CompareType << "\n");

  // Extend or adjust operands *to the determined comparison type*
  auto extendOrAdjustOperand = 
      [&](Value *Operand, Value* OriginalOperand, const std::string &Side) -> Value* {
      if (Operand->getType() == CompareType) {
          return Operand; // Already correct type
      }

      // Check if it's a NonStd ConstantInt needing specific extension
      if (isa<ConstantInt>(Operand) && isNonStandardInt(Operand->getType())) {
          LLVM_DEBUG(dbgs() << Indent << "    Extending NonStd ConstantInt " << Side << ": " << *Operand << "\n");
          if (CmpI->isSigned()) {
              LLVM_DEBUG(dbgs() << Indent << "      using SExt to " << *CompareType << "\n");
              return Builder.CreateSExt(Operand, CompareType, Operand->getName() + ".sext");
          } else {
              LLVM_DEBUG(dbgs() << Indent << "      using ZExt to " << *CompareType << "\n");
              return Builder.CreateZExt(Operand, CompareType, Operand->getName() + ".zext");
          }
      } else {
          // Otherwise, use adjustType for general type mismatches
          LLVM_DEBUG(dbgs() << Indent << "    Adjusting " << Side << ": " << *Operand << " to " << *CompareType << "\n");
          return adjustType(Operand, CompareType, Builder, Indent + "      Adjusting ICmp " + Side + ": ");
      }
  };

  LHS = extendOrAdjustOperand(LHS, OrigLHS, "LHS");
  RHS = extendOrAdjustOperand(RHS, OrigRHS, "RHS");

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
  Value *OrigValue = Store->getValueOperand();
  Value *OrigPtr = Store->getPointerOperand();

  // Get the value being stored (possibly promoted)
  Value *StoredValue = getPromotedValue(OrigValue, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // Get the pointer (we don't normally promote pointers)
  Value *Ptr = getPromotedValue(OrigPtr, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // Determine the type expected by the store based on the pointer
  Type *ExpectedType = Store->getValueOperand()->getType(); // Original value's type
   // More robust way if pointer type changed (though unlikely here):
   // Type *ExpectedType = cast<PointerType>(Ptr->getType())->getElementType();

  // Check if the value is a NonStd ConstantInt needing specific extension
  if (isa<ConstantInt>(StoredValue) && isNonStandardInt(StoredValue->getType())) {
      LLVM_DEBUG(dbgs() << Indent << "    Extending NonStd ConstantInt for Store: " << *StoredValue << "\n");
      // Store doesn't imply signedness, use ZExt before potential truncation
      Value *ExtendedValue = Builder.CreateZExt(StoredValue, PromotedTy, StoredValue->getName() + ".store.zext");
      LLVM_DEBUG(dbgs() << Indent << "      using ZExt to " << *PromotedTy << " -> " << *ExtendedValue << "\n");
      StoredValue = ExtendedValue; // Use the extended value for further adjustment
  }

  // Check if the potentially extended value type needs adjustment to match what's expected by the store
  if (StoredValue->getType() != ExpectedType) {
    // Use adjustType to convert the stored value back to the expected type (e.g., truncate)
    LLVM_DEBUG(dbgs() << Indent << "    Adjusting store value from " << *StoredValue->getType() << " to " << *ExpectedType << "\n");
    StoredValue = adjustType(StoredValue, ExpectedType, Builder, Indent + "      Adjusting store val: ");
  }

  // Create a new store instruction
  StoreInst *NewStore = Builder.CreateStore(StoredValue, Ptr);

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

// Add new handler for BitCastInst
static void processBitCastInst(BitCastInst *BitCast, Type *NonStdType, Type *PromotedTy,
                             IRBuilder<> &Builder, const std::string &Indent,
                             SmallVectorImpl<Replacement> &Replacements,
                             SmallDenseMap<Value *, Value *> &PromotedValues) {
  Value *SrcOp = BitCast->getOperand(0);
  Value *PromotedSrc = getPromotedValue(SrcOp, NonStdType, PromotedTy, Builder, Indent, PromotedValues);
  Type *DestTy = BitCast->getDestTy();
  Type *TargetType = DestTy; // Default to original destination type

  // If the destination type was non-standard, we need to cast to the *promoted* type instead.
  if (isNonStandardInt(DestTy)) {
    TargetType = HipPromoteIntsPass::getPromotedType(DestTy);
     LLVM_DEBUG(dbgs() << Indent << "    Changing bitcast target from non-standard " << *DestTy << " to " << *TargetType << "\n");
  }

  // Adjust the source operand to the target type if necessary *before* creating the new BitCastInst
  PromotedSrc = adjustType(PromotedSrc, TargetType, Builder, Indent + "  Adjusting BitCast Src: ");

  Value *NewInst = Builder.CreateBitCast(PromotedSrc, TargetType);
  LLVM_DEBUG(dbgs() << Indent << "  " << *BitCast << "   promoting BitCast: ====> "
                    << *NewInst << "\n");
  finalizePromotion(BitCast, NewInst, Replacements, PromotedValues);
}

// Add new handler for ExtractElementInst
static void processExtractElementInst(ExtractElementInst *ExtractI, Type *NonStdType, Type *PromotedTy,
                                    IRBuilder<> &Builder, const std::string &Indent,
                                    SmallVectorImpl<Replacement> &Replacements,
                                    SmallDenseMap<Value *, Value *> &PromotedValues) {
  Value *VecOp = ExtractI->getVectorOperand();
  Value *IndexOp = ExtractI->getIndexOperand(); // Index type is usually standard (i32/i64)
  Type *ElementTy = ExtractI->getType(); // The type of the element being extracted

  // Get the potentially promoted vector operand
  // Note: Promoting vectors themselves might be complex. For now, assume getPromotedValue handles it
  // if the *element type* of the vector matches NonStdType. This might need refinement.
  Value *PromotedVecOp = getPromotedValue(VecOp, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  // Determine the target type for the extracted element
  Type* TargetElementType = ElementTy;
  if (isNonStandardInt(ElementTy)) {
      TargetElementType = HipPromoteIntsPass::getPromotedType(ElementTy);
      LLVM_DEBUG(dbgs() << Indent << "    Changing extractelement result type from non-standard " << *ElementTy << " to " << *TargetElementType << "\n");
  }

  // Check if the promoted vector operand's element type matches the target element type
  Type* PromotedVecElementTy = cast<VectorType>(PromotedVecOp->getType())->getElementType();

  // If the promoted vector's element type is different from the target type we need,
  // we might need an intermediate bitcast of the vector or element-wise conversion.
  // For simplicity, let's assume for now that getPromotedValue gave us a vector
  // whose element type matches the *PromotedTy* if the original vector element was NonStdType.
  // If the original element type was standard, PromotedVecOp should be the original vector.
  if (isNonStandardInt(ElementTy) && PromotedVecElementTy != TargetElementType) {
      // This case is complex. How did getPromotedValue handle the vector?
      // Let's adjust the PromotedVecOp *if* its element type matches the global PromotedTy
      if (PromotedVecElementTy == PromotedTy) {
         // We have a vector of PromotedTy, but we need to extract TargetElementType.
         // This usually implies the original extract was non-std -> non-std.
         // We create the extract with PromotedTy elements first.
         Value *ExtractPromoted = Builder.CreateExtractElement(PromotedVecOp, IndexOp);
         // Then adjust the result.
         Value* NewInst = adjustType(ExtractPromoted, TargetElementType, Builder, Indent + "  Adjusting Extracted Element: ");
         LLVM_DEBUG(dbgs() << Indent << "  " << *ExtractI << "   promoting ExtractElement (adjusting result): ====> " << *NewInst << "\n");
         finalizePromotion(ExtractI, NewInst, Replacements, PromotedValues);
         return;
      } else {
        // If PromotedVecElementTy is neither the original ElementTy nor the PromotedTy,
        // this scenario needs more complex handling (potentially element-wise zext/trunc on vector).
        // For now, assert or fallback to simpler logic.
         LLVM_DEBUG(dbgs() << Indent << "WARN: Unhandled case in ExtractElement promotion. Vector element type mismatch." << *PromotedVecOp->getType() << " vs " << *TargetElementType << "\n");
         // Fallback: Create extract with original types (might fail later)
         Value *NewInst = Builder.CreateExtractElement(VecOp, IndexOp);
         finalizePromotion(ExtractI, NewInst, Replacements, PromotedValues);
         return;
      }

  }

  // Create the new ExtractElement instruction using the potentially promoted vector
  // and the target element type.
  Value *NewInst = Builder.CreateExtractElement(PromotedVecOp, IndexOp);

  // Adjust the result if the element type needs changing from what CreateExtractElement produced.
  if (NewInst->getType() != TargetElementType) {
      NewInst = adjustType(NewInst, TargetElementType, Builder, Indent + "  Adjusting Extracted Element Type: ");
  }

  LLVM_DEBUG(dbgs() << Indent << "  " << *ExtractI << "   promoting ExtractElement: ====> " << *NewInst << "\n");
  finalizePromotion(ExtractI, NewInst, Replacements, PromotedValues);
}

// Add new handler for InsertElementInst
static void processInsertElementInst(InsertElementInst *InsertI, Type *NonStdType, Type *PromotedTy,
                                     IRBuilder<> &Builder, const std::string &Indent,
                                     SmallVectorImpl<Replacement> &Replacements,
                                     SmallDenseMap<Value *, Value *> &PromotedValues) {
    Value *VecOp = InsertI->getOperand(0); // The vector being inserted into
    Value *ElementOp = InsertI->getOperand(1); // The element being inserted
    Value *IndexOp = InsertI->getOperand(2); // The index
    Type *ResultVecTy = InsertI->getType(); // Type of the resulting vector
    Type *ElementTy = ElementOp->getType(); // Type of the element being inserted

    assert(isa<VectorType>(ResultVecTy) && "InsertElement should produce a vector type");
    Type *VecElementTy = cast<VectorType>(ResultVecTy)->getElementType();

    // Get potentially promoted operands
    Value *PromotedVecOp = getPromotedValue(VecOp, NonStdType, PromotedTy, Builder, Indent, PromotedValues);
    Value *PromotedElementOp = getPromotedValue(ElementOp, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

    // Determine the target type for the resulting vector and its elements
    Type *TargetVecTy = ResultVecTy;
    Type *TargetElementTy = VecElementTy;

    if (isNonStandardInt(VecElementTy)) {
        TargetElementTy = HipPromoteIntsPass::getPromotedType(VecElementTy);
        TargetVecTy = VectorType::get(TargetElementTy, cast<VectorType>(ResultVecTy)->getElementCount());
        LLVM_DEBUG(dbgs() << Indent << "    Changing insertelement result vector type from non-standard element " << *ResultVecTy << " to " << *TargetVecTy << "\n");
    }

    // Ensure the vector operand has the correct TargetVecTy
    // Note: This assumes getPromotedValue correctly handled vector promotion if necessary.
    // If the original vector had non-std elements, PromotedVecOp should ideally have TargetVecTy.
    if (PromotedVecOp->getType() != TargetVecTy) {
         LLVM_DEBUG(dbgs() << Indent << "    Adjusting InsertElement Vector Operand Type: " << *PromotedVecOp->getType() << " -> " << *TargetVecTy << "\n");
         PromotedVecOp = adjustType(PromotedVecOp, TargetVecTy, Builder, Indent + "      ");
    }

    // Ensure the element operand has the correct TargetElementTy
    if (PromotedElementOp->getType() != TargetElementTy) {
        LLVM_DEBUG(dbgs() << Indent << "    Adjusting InsertElement Element Operand Type: " << *PromotedElementOp->getType() << " -> " << *TargetElementTy << "\n");
        PromotedElementOp = adjustType(PromotedElementOp, TargetElementTy, Builder, Indent + "      ");
    }

    // Create the new InsertElement instruction
    Value *NewInst = Builder.CreateInsertElement(PromotedVecOp, PromotedElementOp, IndexOp, InsertI->getName());

    // Finalize promotion
    LLVM_DEBUG(dbgs() << Indent << "  " << *InsertI << "   promoting InsertElement: ====> " << *NewInst << "\n");
    finalizePromotion(InsertI, NewInst, Replacements, PromotedValues);
}

// Add new handler for SExtInst
static void processSExtInst(SExtInst *SExtI, Type *NonStdType /* Type being promoted, e.g. i56 */,
                            Type *PromotedTy /* Type to promote to, e.g. i64 */,
                            IRBuilder<> &Builder, const std::string &Indent,
                            SmallVectorImpl<Replacement> &Replacements,
                            SmallDenseMap<Value *, Value *> &PromotedValues) {
  Value *SrcOp = SExtI->getOperand(0);
  Type *SrcTy = SrcOp->getType();
  Type *DestTy = SExtI->getDestTy();

  // Get the source value (might be original NonStd ConstantInt)
  Value *PromotedSrc = getPromotedValue(SrcOp, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

  bool IsSrcNonStandard = isNonStandardInt(SrcTy);
  bool IsDestNonStandard = isNonStandardInt(DestTy);
  Type *PromotedDestTy = IsDestNonStandard ? HipPromoteIntsPass::getPromotedType(DestTy) : DestTy;

  Value *NewValue = nullptr;

  // Check if SrcOp is a NonStd ConstantInt needing specific extension
  if (isa<ConstantInt>(PromotedSrc) && IsSrcNonStandard) {
      LLVM_DEBUG(dbgs() << Indent << "  Extending NonStd ConstantInt operand for SExt: " << *PromotedSrc << "\n");
      NewValue = Builder.CreateSExt(PromotedSrc, PromotedDestTy, PromotedSrc->getName() + ".constexpr.sext");
      LLVM_DEBUG(dbgs() << Indent << "  " << *SExtI << "   promoting SExt (non-std const src): ====> " << *NewValue << "\n");
      finalizePromotion(SExtI, NewValue, Replacements, PromotedValues);
      return;
  }

  // Original Logic (slightly adapted)
  if (IsSrcNonStandard) {
    // Case 1: Source is Non-Standard (but not a ConstantInt, handled above)
    // PromotedSrc should now have the PromotedTy (e.g., i64) from recursive processing
    assert(PromotedSrc->getType() == PromotedTy && "Non-standard instruction source operand was not promoted correctly");

    // Adjust the PromotedSrc (which is i64) to match the promoted destination type (PromotedDestTy)
    NewValue = adjustType(PromotedSrc, PromotedDestTy, Builder, Indent + "    Adjusting NonStd Src for SExt: ");
    if (NewValue == PromotedSrc) {
      LLVM_DEBUG(dbgs() << Indent << "  " << *SExtI << "   promoting SExt (non-std inst src, becomes no-op): ====> " << *NewValue << "\n");
    } else {
      LLVM_DEBUG(dbgs() << Indent << "  " << *SExtI << "   promoting SExt (non-std inst src, requires adjust): ====> " << *NewValue << "\n");
    }
    finalizePromotion(SExtI, NewValue, Replacements, PromotedValues);

  } else if (IsDestNonStandard) {
    // Case 2: Destination is Non-Standard (Source is Standard)
    // PromotedSrc should have the original standard source type SrcTy.
    assert(PromotedSrc->getType() == SrcTy && "Promoted source type mismatch for standard SExt source");

    // Create SExt using the standard source (PromotedSrc) to the *promoted* destination type.
    NewValue = Builder.CreateSExt(PromotedSrc, PromotedDestTy); // Use SExt
    LLVM_DEBUG(dbgs() << Indent << "  " << *SExtI << "   promoting SExt (non-std dest): ====> " << *NewValue << "\n");
    finalizePromotion(SExtI, NewValue, Replacements, PromotedValues);

  } else {
    // Case 3: Source and Destination are Standard
    assert(PromotedSrc->getType() == SrcTy && "Promoted source type mismatch for standard SExt source");
    NewValue = Builder.CreateSExt(PromotedSrc, DestTy); // Use SExt
    LLVM_DEBUG(dbgs() << Indent << "  " << *SExtI << "   promoting SExt (standard src/dest): ====> " << *NewValue << "\n");
    finalizePromotion(SExtI, NewValue, Replacements, PromotedValues);
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
  } else if (auto *SExtI = dyn_cast<SExtInst>(I)) { // Add handler for SExtInst
      processSExtInst(SExtI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
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
  } else if (auto *BitCast = dyn_cast<BitCastInst>(I)) { // Add handler for BitCast
      processBitCastInst(BitCast, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *ExtractI = dyn_cast<ExtractElementInst>(I)) { // Add handler for ExtractElement
      processExtractElementInst(ExtractI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *InsertI = dyn_cast<InsertElementInst>(I)) { // Add handler for InsertElement
      processInsertElementInst(InsertI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else {
    llvm::errs() << "HipPromoteIntsPass: Unhandled instruction type: " << I->getOpcodeName() << "\n";
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

    // Create a vector of truncated linked lists of instructions
    std::vector<std::vector<Instruction *>> AllChains;
    for (Instruction *I : WorkList) {
      std::vector<std::vector<Instruction *>> Chains = getLinkedListsFromUseDefChain(I);
      // Process each chain individually
      for (auto &Chain : Chains) {
        std::vector<Instruction *> TruncatedChain = truncateUseDefLL(Chain);
        AllChains.push_back(TruncatedChain);
      }
    }

    // Debug output for all chains
    for (unsigned i = 0; i < AllChains.size(); ++i) {
      LLVM_DEBUG(dbgs() << "Truncated chain " << i << ":\n");
      for (Instruction *I : AllChains[i]) {
        LLVM_DEBUG(dbgs() << "  " << *I << "\n");
      }
    }

    for (std::vector<Instruction *> LL : AllChains) {
      for (Instruction *I : LL) {
        LLVM_DEBUG(dbgs() << "Truncated linked list: " << *I << "\n");
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