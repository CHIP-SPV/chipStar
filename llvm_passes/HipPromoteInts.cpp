#include "HipPromoteInts.h"
#include "llvm/ADT/SmallVector.h" // Include for SmallVector
#include "llvm/ADT/SmallPtrSet.h" // Include for SmallPtrSet
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <functional>
#include <list>
#define DEBUG_TYPE "hip-promote-ints"

// Define the static member

/**
 * @brief Promotes non-standard integer types (e.g., i33, i56) to the next standard width.
 *
 * LLVM's loop optimizations can generate integer types with bit widths that are
 * not powers of two (or 1). These non-standard types can cause issues during
 * later stages, particularly SPIR-V translation.
 * Key Data Structures:
 * -------------------- 
 * - `PromotedValues`: Maps original `Value*` (instructions, constants) to their promoted `Value*` equivalents
 *   within the context of a single `promoteChain` call.
 * - `Replacements`: Stores pairs of `{original instruction, new value}` created during `promoteChain`.
 * - `PendingPhiAdds`: Temporarily stores `{TargetPhi, OriginalValue, IncomingBlock}` tuples for PHI inputs
 *   whose `OriginalValue` hadn't been processed yet when `processPhiNode` was called.
 * Algorithm Overview:
 */
using namespace llvm;

SmallPtrSet<Instruction *, 32> HipPromoteIntsPass::GlobalVisited;
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
/// @return back-pruned linked list
static std::vector<Instruction *> backPruneLL(std::vector<Instruction *> LL){
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
static void getLinkedListsFromUseDefChain(Instruction *I, std::vector<std::vector<Instruction *>> &prunedChains) {
  // Check if the instruction is already in a pruned chain
  for (auto &chain : prunedChains) {
    if (std::find(chain.begin(), chain.end(), I) != chain.end()) {
      // LLVM_DEBUG(dbgs() << "Skipping instruction already in a pruned chain: " << *I << "\n");
      // LLVM_DEBUG(dbgs() << "Found in this chain:\n");
      // for (Instruction *Inst : chain) {
      //   LLVM_DEBUG(dbgs() << "  " << *Inst << "\n");
      // }
      return;
    }
  }

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
    // LLVM_DEBUG(dbgs() << "No chains found for: " << *I << "\n");
    return;
  }

  // Print the linked lists
  // LLVM_DEBUG(dbgs() << "Found " << Chains.size() << " chains for: " << *I << "\n");
  for (unsigned i = 0; i < Chains.size(); ++i) {
    auto CurrentChain = Chains[i];

    // LLVM_DEBUG(dbgs() << "Chain " << i << ":\n");
    // for (Instruction *Inst : CurrentChain) {
    //   LLVM_DEBUG(dbgs() << "  " << *Inst << "\n");
    // }

    // auto prunePossible = false;
    // auto ChainBegin = Chains[i].begin();
    // auto ChainEnd = Chains[i].end();
    // for (auto prunedChain :prunedChains ) {
    //   auto lastPrunedChainInstr = prunedChain.back();
    //   auto found = std::find(ChainBegin, ChainEnd, lastPrunedChainInstr);
    //   if (found != ChainEnd) {
    //     LLVM_DEBUG(dbgs() << "Found pruning candidate for Chain " << i << " via instr: " << *lastPrunedChainInstr << "\n");
    //     prunePossible = true;
    //     LLVM_DEBUG(dbgs() << "Old ChainBegin first instr: " << **ChainBegin << "\n");
    //     ChainBegin = found + 1;
    //     LLVM_DEBUG(dbgs() << "New ChainBegin first instr: " << **ChainBegin << "\n");

    //   }
    // }
    // // If we found a pruning candidate, increment ChainBegin until we found a non-standard instruction
    // if (prunePossible) {
    //   while (ChainBegin != ChainEnd && !isNonStandardInt((*ChainBegin)->getType())) {
    //     ++ChainBegin;
    //   }
    //   if (ChainBegin == ChainEnd) {
    //     LLVM_DEBUG(dbgs() << "Chain " << i << " was completely pruned\n");
    //     continue;
    //   }

    //   CurrentChain = std::vector<Instruction *>(ChainBegin, ChainEnd);
    // }

    // LLVM_DEBUG(dbgs() << "Chain " << i << " after pruning:\n");
    // for (Instruction *Inst : CurrentChain) {
    //   LLVM_DEBUG(dbgs() << "  " << *Inst << "\n");
    // }

    auto frontAndBackPrunedChain = backPruneLL(CurrentChain);
    prunedChains.push_back(frontAndBackPrunedChain);
    // LLVM_DEBUG(dbgs() << "Back-pruned chain " << i << ":\n");
    // for (Instruction *Inst : frontAndBackPrunedChain) {
    //   LLVM_DEBUG(dbgs() << "  " << *Inst << "\n");
    // }
  }
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
  assert(TypeToPromote && "TypeToPromote is nullptr");
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
  
  // Add equality operator for comparison
  bool operator==(const Replacement &Other) const {
    return Old == Other.Old && New == Other.New;
  }
};

// Helper to adjust value V to type TargetTy using Builder
static Value* adjustType(Value *V, Type *TargetTy, IRBuilder<> &Builder, bool NeedsSignedExt = false, const std::string& Indent = "", const std::string& NameSuffix = "") {
    // Handle no-op and constant-int extension/truncation
    if (V->getType() == TargetTy)
        return V;
    if (auto *ConstInt = dyn_cast<ConstantInt>(V)) {
        if (NeedsSignedExt)
            return Builder.CreateSExt(ConstInt, TargetTy, V->getName() + ".sext");
        else
            return Builder.CreateZExt(ConstInt, TargetTy, V->getName() + ".zext");
    }

    // Example 1: Source i33, Target i64
    //   adjustType(%val_i33, i64, builder) -> creates '%zext = zext i33 %val_i33 to i64'
    // Example 2: Source i64, Target i32
    //   adjustType(%val_i64, i32, builder) -> creates '%trunc = trunc i64 %val_i64 to i32'
    // Example 3: Source i64, Target <64 x i1> (Different types, same size)
    //   adjustType(%val_i64, <64 x i1>, builder) -> creates '%bitcast = bitcast i64 %val_i64 to <64 x i1>'

    unsigned SrcBits = V->getType()->getPrimitiveSizeInBits();
    unsigned DstBits = TargetTy->getPrimitiveSizeInBits();

    LLVM_DEBUG(dbgs() << Indent << "Adjusting type of " << *V << " from " << *V->getType() << " to " << *TargetTy << "\n");

    Value* AdjustedV = nullptr;
    std::string Name = NameSuffix.empty() ? "" : V->getName().str() + NameSuffix;
    
    if (DstBits < SrcBits) {
        AdjustedV = Builder.CreateTrunc(V, TargetTy, Name);
        LLVM_DEBUG(dbgs() << Indent << "  Created Trunc: " << *AdjustedV << "\n");
    } else if (DstBits > SrcBits) {
        AdjustedV = Builder.CreateZExt(V, TargetTy, Name);
        LLVM_DEBUG(dbgs() << Indent << "  Created ZExt: " << *AdjustedV << "\n");
    } else {
        AdjustedV = Builder.CreateBitCast(V, TargetTy, Name);
        LLVM_DEBUG(dbgs() << Indent << "  Created BitCast: " << *AdjustedV << "\n");
    }
    if (AdjustedV) HipPromoteIntsPass::GlobalVisited.insert(dyn_cast<Instruction>(AdjustedV));
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
  LLVM_DEBUG(dbgs() << Indent << "    getPromotedValue for: " << *V << " NonStdType: " << *NonStdType << " PromotedTy: " << *PromotedTy << "\n");

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
    Value *NewV = adjustType(V, PromotedTy, Builder, false, Indent + "      ");

    PromotedValues[V] = NewV;
    return NewV;
  }

  // Otherwise return original value
  LLVM_DEBUG(dbgs() << Indent << "      Using original value: " << *V << "\n");
  return V;
};

void addReplacement(Instruction *Old, Value *New, SmallVectorImpl<Replacement> &Replacements) {
  LLVM_DEBUG(dbgs() << "addReplacement: " << *Old << " with " << *New << "\n");
  // assert that none of the entries in Replacements have Old as an operand
  for (auto &R : Replacements) {
    if (R.Old == Old && R.New == New) return; // already exists
    if (R.Old == Old) assert(R.New!= New && "Changing old instruction to different value");
  }
  Replacements.push_back(Replacement(Old, New));
}

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
    addReplacement(Old, New, Replacements);
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
static Value *processPhiNode(PHINode *Phi, Type *NonStdType, Type *PromotedTy,
                           IRBuilder<> &Builder, const std::string &Indent,
                           SmallVectorImpl<Replacement> &Replacements,
                           SmallDenseMap<Value *, Value *> &PromotedValues) {
  // Create new PHI node with the promoted type
  Type *PromotedType = HipPromoteIntsPass::getPromotedType(Phi->getType());
  
    // Check if the incoming values are already promoted
    for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
      Value *OriginalValue = Phi->getIncomingValue(i);
      LLVM_DEBUG(dbgs() << Indent << "    Processing incoming: " << *OriginalValue << "\n");
      if(!PromotedValues.count(OriginalValue)) {
        LLVM_DEBUG(dbgs() << Indent << "      Incoming value not yet promoted, deferring: " << *OriginalValue << "\n");
        return nullptr;
      }
  }

      
  // Process incoming values
  PHINode *NewPhi = PHINode::Create(PromotedType, Phi->getNumIncomingValues(), "", Phi);
  unsigned PendingCount = 0;
  for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
      Value *OriginalValue = Phi->getIncomingValue(i);
      BasicBlock *IncomingBlock = Phi->getIncomingBlock(i);
      Value *NewIncomingValue = nullptr;
      assert(PromotedValues.count(OriginalValue) && "Incoming value not yet promoted");
      NewIncomingValue = PromotedValues[OriginalValue];
      // Ensure the determined value matches the NewPhi's type
      LLVM_DEBUG(dbgs() << Indent << "      Adjusting incoming value type if needed...\n");
      Value *AdjustedValue = adjustType(NewIncomingValue, PromotedType, Builder, false, Indent + "        ");
      LLVM_DEBUG(dbgs() << Indent << "      Adding incoming: [" << *AdjustedValue << ", " << IncomingBlock->getName() << "]\n");
      NewPhi->addIncoming(AdjustedValue, IncomingBlock);
  }
  
  PromotedValues[Phi] = NewPhi;
  LLVM_DEBUG(dbgs() << Indent << "  " << *Phi
                    << "   promoting PHI node: ====> " << *NewPhi << " ("
                    << NewPhi->getNumIncomingValues() << " initial incoming, "
                    << PendingCount << " pending)\n");

  addReplacement(Phi, NewPhi, Replacements);
  return NewPhi;
}

// Refined processZExtInst logic:
static Value *processZExtInst(ZExtInst *ZExtI, Type *NonStdType /* Type being promoted, e.g. i56 */,
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
      NewValue = adjustType(PromotedSrc, PromotedDestTy, Builder, false, Indent, ".constexpr.zext");
      LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (non-std const src): ====> " << *NewValue << "\n");
      finalizePromotion(ZExtI, NewValue, Replacements, PromotedValues);
      return NewValue;
  }

  if (IsSrcNonStandard) {
    // Case 1: Source is Non-Standard (e.g., zext i56 -> i64) (but not a ConstantInt, handled above)
    // PromotedSrc should now have the PromotedTy (e.g., i64)
    assert(PromotedSrc->getType() == PromotedTy && "Non-standard source operand was not promoted correctly in getPromotedValue");

    // Adjust the PromotedSrc (which is i64) to match the promoted destination type (PromotedDestTy)
    NewValue = adjustType(PromotedSrc, PromotedDestTy, Builder, false, Indent + "    Adjusting NonStd Src for ZExt: ");
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
    NewValue = adjustType(PromotedSrc, PromotedDestTy, Builder, false, Indent);
    LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (non-std dest): ====> " << *NewValue << "\n");
    finalizePromotion(ZExtI, NewValue, Replacements, PromotedValues);

  } else {
    // Case 3: Source and Destination are Standard (e.g., zext i32 -> i64)
    // Recreate the instruction. Since the source is standard, getPromotedValue
    // should have returned a value with the original source type SrcTy.
    assert(PromotedSrc->getType() == SrcTy && "Promoted source type mismatch for standard ZExt source");

    // Create the ZExt with the standard source (PromotedSrc) and original standard destination type.
    NewValue = adjustType(PromotedSrc, DestTy, Builder, false, Indent);
    LLVM_DEBUG(dbgs() << Indent << "  " << *ZExtI << "   promoting ZExt (standard src/dest): ====> " << *NewValue << "\n");
    // Map original ZExt result to new ZExt result. Both should have same standard type.
    finalizePromotion(ZExtI, NewValue, Replacements, PromotedValues);
  }
  return NewValue;
}


static Value *processTruncInst(TruncInst *TruncI, Type *NonStdType, Type *PromotedTy,
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
              << "   processing truncation from standard to non-standard.\n");

    // Determine the promoted type for the non-standard destination
    Type *PromotedDestTy = HipPromoteIntsPass::getPromotedType(DestTy);
    LLVM_DEBUG(dbgs() << Indent << "      Promoted destination type: " << *PromotedDestTy << "\n");

    // Get the source operand (should be standard type)
    // getPromotedValue should return the original standard source operand here
    Value *StandardSrc = getPromotedValue(SrcOp, NonStdType, PromotedTy, Builder, Indent, PromotedValues);
    assert(StandardSrc == SrcOp && "Expected getPromotedValue to return original standard source");
    assert(StandardSrc->getType() == SrcOp->getType() && "Source type mismatch");

    // Adjust the standard source to the *promoted destination type*
    Value* AdjustedSrc = adjustType(StandardSrc, PromotedDestTy, Builder, false, Indent + "      Adjusting std src to promoted dest type: ");
    LLVM_DEBUG(dbgs() << Indent << "      Adjusted source value: " << *AdjustedSrc << "\n");

    // Map the original trunc instruction to this adjusted source value
    PromotedValues[TruncI] = AdjustedSrc;
    LLVM_DEBUG(dbgs() << Indent << "      Mapped original trunc " << *TruncI << " to " << *AdjustedSrc << " in PromotedValues\n");

    // Replace the original instruction with the adjusted source value.
    // Users outside this promotion chain will use this adjusted value.
    addReplacement(TruncI, AdjustedSrc, Replacements);
    LLVM_DEBUG(dbgs() << Indent << "      Scheduled replacement of " << *TruncI << " with " << *AdjustedSrc << "\n");
    return AdjustedSrc;
  }

  // --- Original logic for other truncation cases ---
  LLVM_DEBUG(dbgs() << Indent << "  Processing non-(standard->non-standard) truncation: " << *TruncI << "\n");

  // Verify the source is actually of our promoted type
  if (PromotedSrc->getType() != PromotedTy) {
    // Use adjustType to ensure the source matches the promoted type
    PromotedSrc = adjustType(PromotedSrc, PromotedTy, Builder, false, Indent + "    Adjusting source to promoted type: ");
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
  Value *NewTrunc = adjustType(PromotedSrc, DestTy, Builder, false, Indent + "  Creating external trunc: ");
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

  addReplacement(TruncI, NewTrunc, Replacements);
  return NewTrunc;
}

static Value *processBinaryOperator(BinaryOperator *BinOp, Type *NonStdType, Type *PromotedTy,
                                  IRBuilder<> &Builder, const std::string &Indent,
                                  SmallVectorImpl<Replacement> &Replacements,
                                  SmallDenseMap<Value *, Value *> &PromotedValues) {
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
  // Later, adjustType will see RHS is i33 5 and create a 'zext i33 5 to i64' for the add.
  //
  // Example 2: %orig_sub = sub i64 %std_val, i64 %promoted_non_std
  //   OrigLHS = %std_val (i64)
  //   OrigRHS = %promoted_non_std (i64, already promoted from i33 earlier)
  // After getPromotedValue:
  //   LHS = %std_val (already i64)
  //   RHS = %promoted_non_std (already i64)
  // Later, adjustType will see types match TargetType (i64) and do nothing.
  //
  // Example 3: %orig_sdiv = sdiv i33 %another_inst, i33 -1
  //   OrigLHS = %another_inst (i33)
  //   OrigRHS = i33 -1 (ConstantInt)
  // After getPromotedValue:
  //   LHS = result of adjustType, e.g., '%another_inst.zext = zext i33 %another_inst to i64'
  //   RHS = original ConstantInt* for i33 -1
  // Later, adjustType will see RHS is i33 -1 and the opcode is SDiv,
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
  bool NeedsSExt = (BinOp->getOpcode() == Instruction::SDiv ||
                    BinOp->getOpcode() == Instruction::SRem ||
                    BinOp->getOpcode() == Instruction::AShr);
  LHS = adjustType(LHS, TargetType, Builder, NeedsSExt, Indent + "      ");
  RHS = adjustType(RHS, TargetType, Builder, NeedsSExt, Indent + "      ");

  // Now LHS and RHS must have the same type
  assert(LHS->getType() == RHS->getType() && "Operand types mismatch for BinOp after adjustment!");
  Value *NewInst = Builder.CreateBinOp(BinOp->getOpcode(), LHS, RHS);

  LLVM_DEBUG(dbgs() << Indent << "  " << *BinOp << "   promoting BinOp: ====> "
                    << *NewInst << "\n");
  finalizePromotion(BinOp, NewInst, Replacements, PromotedValues);
  return NewInst;
}

static Value *processSelectInst(SelectInst *SelI, Type *NonStdType, Type *PromotedTy,
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
             Condition = adjustType(Condition, ConditionPromotedTy, Builder, false, Indent + "      ", ".zext");
             LLVM_DEBUG(dbgs() << Indent << "      Created ZExt via adjustType: " << *Condition << "\n");
        } else {
            Condition = adjustType(Condition, ConditionPromotedTy, Builder, false, Indent + "  Adjusting NonStd Cond: ");
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
  TrueVal  = adjustType(TrueVal,  TargetType, Builder, false, Indent + "      ");
  FalseVal = adjustType(FalseVal, TargetType, Builder, false, Indent + "      ");

  // Now TrueVal and FalseVal must have the same type
  assert(TrueVal->getType() == FalseVal->getType() && "Operand types mismatch for Select after adjustment!");
  Value *NewSelect = Builder.CreateSelect(Condition, TrueVal, FalseVal, SelI->getName());

  LLVM_DEBUG(dbgs() << Indent << "  " << *SelI << "   promoting Select: ====> "
                    << *NewSelect << "\n");
  finalizePromotion(SelI, NewSelect, Replacements, PromotedValues);
  return NewSelect;
}

static Value *processICmpInst(ICmpInst *CmpI, Type *NonStdType, Type *PromotedTy,
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
  LHS = adjustType(LHS, CompareType, Builder, CmpI->isSigned(), Indent + "      ");
  RHS = adjustType(RHS, CompareType, Builder, CmpI->isSigned(), Indent + "      ");

  // Now LHS and RHS must have the same type
  assert(LHS->getType() == RHS->getType() && "Operand types mismatch for ICmp after adjustment!");
  // Create new comparison instruction (result is always i1)
  Value *NewCmp = Builder.CreateICmp(CmpI->getPredicate(), LHS, RHS, CmpI->getName());

  LLVM_DEBUG(dbgs() << Indent << "  " << *CmpI << "   promoting ICmp: ====> "
                    << *NewCmp << "\n");
  finalizePromotion(CmpI, NewCmp, Replacements, PromotedValues);
  return NewCmp;
}

static Value *processCallInst(CallInst *OldCall, Type *NonStdType, Type *PromotedTy,
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
      NewArg = adjustType(NewArg, ExpectedType, Builder, false, Indent + "      Adjusting arg: ");
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
  return NewCall;
}

static Value *processStoreInst(StoreInst *Store, Type *NonStdType, Type *PromotedTy,
                             IRBuilder<> &Builder, const std::string &Indent,
                             SmallVectorImpl<Replacement> &Replacements,
                             SmallDenseMap<Value *, Value *> &PromotedValues) {
  Value *OrigValue = Store->getValueOperand();
  Value *OrigPtr = Store->getPointerOperand();

  // Promote store of non-standard integer types by bitcasting pointer and storing promoted type.
  if (auto *ValIntTy = dyn_cast<IntegerType>(OrigValue->getType())) {
    unsigned BW = ValIntTy->getBitWidth();
    if (!HipPromoteIntsPass::isStandardBitWidth(BW)) {
      LLVM_DEBUG(dbgs() << Indent << "Promoting store of non-standard i" << BW << "\n");
      // Determine the promoted integer type and address space
      unsigned NewBW = HipPromoteIntsPass::getPromotedBitWidth(BW);
      LLVMContext &Ctx = Store->getContext();
      Type *PromTy = Type::getIntNTy(Ctx, NewBW);
      unsigned AS = OrigPtr->getType()->getPointerAddressSpace();
      PointerType *NewPtrTy = PointerType::get(PromTy, AS);
      // Bitcast the pointer to the promoted pointer type
      Value *CastPtr = Builder.CreateBitCast(OrigPtr, NewPtrTy, Store->getName() + ".promote_ptr");
      // Obtain the promoted value (zero/sign extension handled by getPromotedValue)
      Value *PromValue = getPromotedValue(OrigValue, OrigValue->getType(), PromTy, Builder, Indent, PromotedValues);
      // Create the promoted store
      StoreInst *NewStore = Builder.CreateStore(PromValue, CastPtr);
      NewStore->setAlignment(Store->getAlign());
      NewStore->setVolatile(Store->isVolatile());
      NewStore->setOrdering(Store->getOrdering());
      NewStore->setSyncScopeID(Store->getSyncScopeID());
      addReplacement(Store, NewStore, Replacements);
      return NewStore;
    }
  }

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
      Value *ExtendedValue = adjustType(StoredValue, PromotedTy, Builder, false, Indent + "      ", ".store.zext");
      LLVM_DEBUG(dbgs() << Indent << "      using ZExt to " << *PromotedTy << " -> " << *ExtendedValue << "\n");
      StoredValue = ExtendedValue; // Use the extended value for further adjustment
  }

  // Check if the potentially extended value type needs adjustment to match what's expected by the store
  if (StoredValue->getType() != ExpectedType) {
    // Use adjustType to convert the stored value back to the expected type (e.g., truncate)
    LLVM_DEBUG(dbgs() << Indent << "    Adjusting store value from " << *StoredValue->getType() << " to " << *ExpectedType << "\n");
    StoredValue = adjustType(StoredValue, ExpectedType, Builder, false, Indent + "      Adjusting store val: ");
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
  addReplacement(Store, NewStore, Replacements);
  return NewStore;
}


static Value *processLoadInst(LoadInst *Load, Type *NonStdType, Type *PromotedTy,
                              IRBuilder<> &Builder, const std::string &Indent,
                              SmallVectorImpl<Replacement> &Replacements,
                              SmallDenseMap<Value *, Value *> &PromotedValues) {
  Value *Ptr = Load->getPointerOperand();
  // Promote non-standard integer loads by casting pointer and loading the promoted type.
  if (auto *IntTy = dyn_cast<IntegerType>(Load->getType())) {
    unsigned BW = IntTy->getBitWidth();
    if (!HipPromoteIntsPass::isStandardBitWidth(BW)) {
      LLVM_DEBUG(dbgs() << Indent << "Promoting load of non-standard i" << BW << "\n");
      LLVMContext &Ctx = Load->getContext();
      unsigned NewBW = HipPromoteIntsPass::getPromotedBitWidth(BW);
      Type *NewIntTy = Type::getIntNTy(Ctx, NewBW);
      // Bitcast the pointer to point to the promoted integer type
      PointerType *NewPtrTy = PointerType::get(NewIntTy, 
                                  Ptr->getType()->getPointerAddressSpace());
      Value *CastPtr = Builder.CreateBitCast(Ptr, NewPtrTy, Load->getName() + ".promote_ptr");
      // Create the promoted load
      LoadInst *NewLoad = Builder.CreateLoad(NewIntTy, CastPtr, Load->getName() + ".promote");
      NewLoad->setAlignment(Load->getAlign());
      NewLoad->setVolatile(Load->isVolatile());
      NewLoad->setOrdering(Load->getOrdering());
      NewLoad->setSyncScopeID(Load->getSyncScopeID());
      PromotedValues[Load] = NewLoad;
      addReplacement(Load, NewLoad, Replacements);
      return NewLoad;
    }
  }
  // Fallback: standard-width load
  LoadInst *NewLoad = Builder.CreateLoad(Load->getType(), Ptr, Load->getName());
  NewLoad->setAlignment(Load->getAlign());
  NewLoad->setVolatile(Load->isVolatile());
  NewLoad->setOrdering(Load->getOrdering());
  NewLoad->setSyncScopeID(Load->getSyncScopeID());
  PromotedValues[Load] = NewLoad;
  addReplacement(Load, NewLoad, Replacements);
  return NewLoad;
}

static Value *processReturnInst(ReturnInst *RetI, Type *NonStdType, Type *PromotedTy,
                              IRBuilder<> &Builder, const std::string &Indent,
                              SmallVectorImpl<Replacement> &Replacements,
                              SmallDenseMap<Value *, Value *> &PromotedValues) {
  ReturnInst *NewRet = nullptr;
   // If there's a return value, check if it needs to be promoted/adjusted
  if (RetI->getNumOperands() > 0) {
    Value *RetVal = RetI->getReturnValue();
    Value *NewRetVal = getPromotedValue(RetVal, NonStdType, PromotedTy, Builder, Indent, PromotedValues);

    // Make sure the return value matches the function's return type
    Type *FuncRetType = RetI->getFunction()->getReturnType();
    if (NewRetVal->getType() != FuncRetType) {
      // Use adjustType to match the function's return type
      LLVM_DEBUG(dbgs() << Indent << "    Adjusting return value to match function type\n");
      NewRetVal = adjustType(NewRetVal, FuncRetType, Builder, false, Indent + "      Adjusting ret val: ");
    }

    // Create a new return instruction with the correctly typed value
    NewRet = Builder.CreateRet(NewRetVal);

    LLVM_DEBUG(dbgs() << Indent << "  " << *RetI << "   promoting Return: ====> "
                      << *NewRet << "\n");
    // Return instructions don't produce a value, so we don't put them in PromotedValues
    addReplacement(RetI, NewRet, Replacements);
  } else {
    // Handle void return
    NewRet = Builder.CreateRetVoid();
    LLVM_DEBUG(dbgs() << Indent << "  " << *RetI << "   promoting Return: ====> "
                      << *NewRet << "\n");
    // Return instructions don't produce a value, so we don't put them in PromotedValues
    addReplacement(RetI, NewRet, Replacements);
  }
  return NewRet;
}

// Add new handler for BitCastInst
static Value *processBitCastInst(BitCastInst *BitCast, Type *NonStdType, Type *PromotedTy,
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
  PromotedSrc = adjustType(PromotedSrc, TargetType, Builder, false, Indent + "  Adjusting BitCast Src: ");

  Value *NewInst = Builder.CreateBitCast(PromotedSrc, TargetType);
  LLVM_DEBUG(dbgs() << Indent << "  " << *BitCast << "   promoting BitCast: ====> "
                    << *NewInst << "\n");
  finalizePromotion(BitCast, NewInst, Replacements, PromotedValues);
  return NewInst;
}

// Add new handler for ExtractElementInst
static Value *processExtractElementInst(ExtractElementInst *ExtractI, Type *NonStdType, Type *PromotedTy,
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
         Value* NewInst = adjustType(ExtractPromoted, TargetElementType, Builder, false, Indent + "  Adjusting Extracted Element: ");
         LLVM_DEBUG(dbgs() << Indent << "  " << *ExtractI << "   promoting ExtractElement (adjusting result): ====> " << *NewInst << "\n");
         finalizePromotion(ExtractI, NewInst, Replacements, PromotedValues);
         return NewInst;
      } else {
        // If PromotedVecElementTy is neither the original ElementTy nor the PromotedTy,
        // this scenario needs more complex handling (potentially element-wise zext/trunc on vector).
        // For now, assert or fallback to simpler logic.
         LLVM_DEBUG(dbgs() << Indent << "WARN: Unhandled case in ExtractElement promotion. Vector element type mismatch." << *PromotedVecOp->getType() << " vs " << *TargetElementType << "\n");
         // Fallback: Create extract with original types (might fail later)
         Value *NewInst = Builder.CreateExtractElement(VecOp, IndexOp);
         finalizePromotion(ExtractI, NewInst, Replacements, PromotedValues);
         return NewInst;
      }

  }

  // Create the new ExtractElement instruction using the potentially promoted vector
  // and the target element type.
  Value *NewInst = Builder.CreateExtractElement(PromotedVecOp, IndexOp);

  // Adjust the result if the element type needs changing from what CreateExtractElement produced.
  if (NewInst->getType() != TargetElementType) {
      NewInst = adjustType(NewInst, TargetElementType, Builder, false, Indent + "  Adjusting Extracted Element Type: ");
  }

  LLVM_DEBUG(dbgs() << Indent << "  " << *ExtractI << "   promoting ExtractElement: ====> " << *NewInst << "\n");
  finalizePromotion(ExtractI, NewInst, Replacements, PromotedValues);
  return NewInst;
}

// Add new handler for InsertElementInst
static Value *processInsertElementInst(InsertElementInst *InsertI, Type *NonStdType, Type *PromotedTy,
                                     IRBuilder<> &Builder, const std::string &Indent,
                                     SmallVectorImpl<Replacement> &Replacements,
                                     SmallDenseMap<Value *, Value *> &PromotedValues) {
    Value *VecOp = InsertI->getOperand(0); // The vector being inserted into
    Value *ElementOp = InsertI->getOperand(1); // The element being inserted
    Value *IndexOp = InsertI->getOperand(2); // The index
    Type *ResultVecTy = InsertI->getType(); // Type of the resulting vector

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
         PromotedVecOp = adjustType(PromotedVecOp, TargetVecTy, Builder, false, Indent + "      ");
    }

    // Ensure the element operand has the correct TargetElementTy
    if (PromotedElementOp->getType() != TargetElementTy) {
        LLVM_DEBUG(dbgs() << Indent << "    Adjusting InsertElement Element Operand Type: " << *PromotedElementOp->getType() << " -> " << *TargetElementTy << "\n");
        PromotedElementOp = adjustType(PromotedElementOp, TargetElementTy, Builder, false, Indent + "      ");
    }

    // Create the new InsertElement instruction
    Value *NewInst = Builder.CreateInsertElement(PromotedVecOp, PromotedElementOp, IndexOp, InsertI->getName());

    // Finalize promotion
    LLVM_DEBUG(dbgs() << Indent << "  " << *InsertI << "   promoting InsertElement: ====> " << *NewInst << "\n");
    finalizePromotion(InsertI, NewInst, Replacements, PromotedValues);
    return NewInst;
}

// Add new handler for SExtInst
static Value *processSExtInst(SExtInst *SExtI, Type *NonStdType /* Type being promoted, e.g. i56 */,
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
      NewValue = adjustType(PromotedSrc, PromotedDestTy, Builder, true, Indent, ".constexpr.sext"); // Note: NeedsSignedExt = true
      LLVM_DEBUG(dbgs() << Indent << "  " << *SExtI << "   promoting SExt (non-std const src): ====> " << *NewValue << "\n");
      finalizePromotion(SExtI, NewValue, Replacements, PromotedValues);
      return NewValue;
  }

  // Original Logic (slightly adapted)
  if (IsSrcNonStandard) {
    // Case 1: Source is Non-Standard (but not a ConstantInt, handled above)
    // PromotedSrc should now have the PromotedTy (e.g., i64) from recursive processing
    assert(PromotedSrc->getType() == PromotedTy && "Non-standard instruction source operand was not promoted correctly");

    // Adjust the PromotedSrc (which is i64) to match the promoted destination type (PromotedDestTy)
    LLVM_DEBUG(dbgs() << Indent << "    Adjusting NonStd Src for SExt: " << *PromotedSrc << " to " << *PromotedDestTy << "\n");
    NewValue = adjustType(PromotedSrc, PromotedDestTy, Builder, true, Indent + "    Adjusting NonStd Src for SExt: "); // Note: NeedsSignedExt = true
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
    LLVM_DEBUG(dbgs() << Indent << "    Adjusting NonStd Dest for SExt: " << *PromotedSrc << " to " << *PromotedDestTy << "\n");
    NewValue = adjustType(PromotedSrc, PromotedDestTy, Builder, true, Indent); // Note: NeedsSignedExt = true
    LLVM_DEBUG(dbgs() << Indent << "  " << *SExtI << "   promoting SExt (non-std dest): ====> " << *NewValue << "\n");
    finalizePromotion(SExtI, NewValue, Replacements, PromotedValues);

  } else {
    // Case 3: Source and Destination are Standard
    assert(PromotedSrc->getType() == SrcTy && "Promoted source type mismatch for standard SExt source");
    LLVM_DEBUG(dbgs() << Indent << "    Creating SExt: " << *PromotedSrc << " to " << *DestTy << "\n");
    NewValue = Builder.CreateSExt(PromotedSrc, DestTy); // Use SExt
    LLVM_DEBUG(dbgs() << Indent << "  " << *SExtI << "   promoting SExt (standard src/dest): ====> " << *NewValue << "\n");
    finalizePromotion(SExtI, NewValue, Replacements, PromotedValues);
  }
  return NewValue;
}

Value*processInstruction(Instruction *I, Type *NonStdType, Type *PromotedTy,
                        const std::string &Indent,
                        SmallVectorImpl<Replacement> &Replacements,
                        SmallDenseMap<Value *, Value *> &PromotedValues) {
  IRBuilder<> Builder(I);
  Value *Result = nullptr;

  // Dispatch to the appropriate handler based on instruction type
  if (auto *Phi = dyn_cast<PHINode>(I)) {
      Result = processPhiNode(Phi, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *ZExtI = dyn_cast<ZExtInst>(I)) {
      Result = processZExtInst(ZExtI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *SExtI = dyn_cast<SExtInst>(I)) { // Add handler for SExtInst
      Result = processSExtInst(SExtI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *TruncI = dyn_cast<TruncInst>(I)) {
      Result = processTruncInst(TruncI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *BinOp = dyn_cast<BinaryOperator>(I)) {
      Result = processBinaryOperator(BinOp, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *SelI = dyn_cast<SelectInst>(I)) {
      Result = processSelectInst(SelI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *CmpI = dyn_cast<ICmpInst>(I)) {
      Result = processICmpInst(CmpI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *OldCall = dyn_cast<CallInst>(I)) {
      Result = processCallInst(OldCall, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *Store = dyn_cast<StoreInst>(I)) {
      Result = processStoreInst(Store, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *Load = dyn_cast<LoadInst>(I)) {
      Result = processLoadInst(Load, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *RetI = dyn_cast<ReturnInst>(I)) {
      Result = processReturnInst(RetI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *BitCast = dyn_cast<BitCastInst>(I)) { // Add handler for BitCast
      Result = processBitCastInst(BitCast, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *ExtractI = dyn_cast<ExtractElementInst>(I)) { // Add handler for ExtractElement
      Result = processExtractElementInst(ExtractI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else if (auto *InsertI = dyn_cast<InsertElementInst>(I)) { // Add handler for InsertElement
      Result = processInsertElementInst(InsertI, NonStdType, PromotedTy, Builder, Indent, Replacements, PromotedValues);
  } else {
    llvm::errs() << "HipPromoteIntsPass: Unhandled instruction type: " << I->getOpcodeName() << "\n";
    assert(false && "HipPromoteIntsPass: Unhandled instruction type");
  }
  
  return Result;
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
    std::vector<std::vector<Instruction *>> prunedChains;
    for (Instruction *I : WorkList)
      getLinkedListsFromUseDefChain(I, prunedChains);

    for (unsigned i = 0; i < prunedChains.size(); ++i) {
      LLVM_DEBUG(dbgs() << "Pruned chain " << i << ":\n");
      for (Instruction *Inst : prunedChains[i]) {
        LLVM_DEBUG(dbgs() << "  " << *Inst << "\n");
      }
    }

    auto longestChainNumInstructions = 0;
    for (auto &chain : prunedChains) {
      if (chain.size() > longestChainNumInstructions) {
        longestChainNumInstructions = chain.size();
      }
    }

    SmallVector<Replacement, 16> Replacements;
    SmallDenseMap<Value *, Value *> PromotedValues;
    std::list<Instruction *> DeferredInstructions;
    for (unsigned i = 0; i < longestChainNumInstructions; i++) { // loop over instructions until no more
      for (auto &chain : prunedChains) {
        if (i >= chain.size()) continue;
        auto I = chain[i];

        LLVM_DEBUG(dbgs() << "Processing instruction: " << *I << "\n");
        // Determine NonStdType and PromotedTy for the instruction I
        Type* NonStdType = isNonStandardInt(I->getType()) ? I->getType() : nullptr;
        for (Value *Op : I->operands()) {
          if (isNonStandardInt(Op->getType())) {
            NonStdType = Op->getType();
            break;
          }
        }
        if (!NonStdType) {
          LLVM_DEBUG(dbgs() << "Instruction " << *I << " is standard type. Marking as visited & skipping.\n");
          GlobalVisited.insert(I);
          continue;
        }
        Type* PromotedTy = HipPromoteIntsPass::getPromotedType(NonStdType);
        std::string Indent = ""; // Basic indent for now
        auto processed = processInstruction(I, NonStdType, PromotedTy, Indent, Replacements, PromotedValues);
        if (!processed) {
          LLVM_DEBUG(dbgs() << "Deferring instruction: " << *I << "\n");
          // push if not already in DeferredInstructions
          if (std::find(DeferredInstructions.begin(), DeferredInstructions.end(), I) == DeferredInstructions.end()) {
            DeferredInstructions.push_back(I);
          }
        } else {
          // Mark this instruction as visited
          GlobalVisited.insert(I);

          // remove from DeferredInstructions if it's in there
          if (std::find(DeferredInstructions.begin(), DeferredInstructions.end(), I) != DeferredInstructions.end()) {
            DeferredInstructions.remove(I);
          }
        }

      } // End for (Instruction *I : chain)
    } // End for (auto &chain : prunedChains)

    LLVM_DEBUG(dbgs() << "Processing deferred instructions\n");
    while (!DeferredInstructions.empty()) {
      auto *I = DeferredInstructions.front();
      LLVM_DEBUG(dbgs() << "Processing deferred instruction: " << *I << "\n");
      Type* NonStdType = isNonStandardInt(I->getType()) ? I->getType() : nullptr;
      Type* PromotedTy = HipPromoteIntsPass::getPromotedType(NonStdType);
      std::string Indent = ""; // Basic indent for now
      auto processed = processInstruction(I, NonStdType, PromotedTy, Indent, Replacements, PromotedValues);
      if (!processed) {
        LLVM_DEBUG(dbgs() << "Deferring instruction: " << *I << "\n");
        DeferredInstructions.push_back(I);
      } else {
        // Mark this instruction as visited
        GlobalVisited.insert(I);
      }
      // Don't add duplicate replacements for successfully processed deferred instructions
      if (!GlobalVisited.count(I)) {
        addReplacement(I, processed, Replacements);
        // Mark the instruction as visited right after adding it to Replacements
        GlobalVisited.insert(I);
      }
      DeferredInstructions.pop_front();
      LLVM_DEBUG(dbgs() << "Successfully processed deferred instruction: " << *I << "\n");
    }

    // Now perform the main replacements
    LLVM_DEBUG(dbgs() << "\n\n\n\n\nPerforming main instruction replacements...\n");
    
    // Remove duplicates from Replacements
    std::sort(Replacements.begin(), Replacements.end(), 
              [](const Replacement &A, const Replacement &B) { 
                return std::make_pair(A.Old, A.New) < std::make_pair(B.Old, B.New); 
              });
    Replacements.erase(std::unique(Replacements.begin(), Replacements.end()), Replacements.end());

    
    // Track which instructions have been replaced to avoid duplicates
    SmallPtrSet<Instruction*, 32> ReplacedInstructions;
    
    for (const auto &R : Replacements) {
      // Skip if we've already processed this instruction
      if (!ReplacedInstructions.insert(R.Old).second) {
        LLVM_DEBUG(dbgs() << "Skipping duplicate replacement for: " << *R.Old << " with " << *R.New << "\n");
        continue;
      }
      
      LLVM_DEBUG(dbgs() << "Replacing uses of: " << *R.Old << "    with: " << *R.New << "\n");
      // Make a copy of the users to avoid iterator invalidation
      SmallVector<User*, 8> Users(R.Old->user_begin(), R.Old->user_end());
      for (User *U : Users) {
          // Replace uses only if the user instruction itself is part of the processed chain
          // or if it's not an instruction (e.g. a constant expression using the old value).
          LLVM_DEBUG(dbgs() << "  Updating use in: " << *U << "\n");
          U->replaceUsesOfWith(R.Old, R.New);
      }
    }

    std::set<Instruction *> InstructionsToDelete;
    for (auto &R : Replacements) {
      LLVM_DEBUG(dbgs() << "Will delete instruction: " << *R.Old << " which is replaced by " << *R.New << "\n");
      InstructionsToDelete.insert(R.Old);
    }


    // Finally, delete the original instructions in reverse order to handle dependencies
    for (auto It = InstructionsToDelete.rbegin(); It != InstructionsToDelete.rend(); ++It) {
      LLVM_DEBUG(dbgs() << "Deleting instruction: " << **It << "\n");
      if (!(*It)->use_empty()) {
        LLVM_DEBUG(dbgs() << "WARNING: Instruction still has uses before deletion: " << **It << "\n");
        for (User *U : (*It)->users()) {
          LLVM_DEBUG(dbgs() << "  User: " << *U << "\n");
        }
        // Force replacement again just in case
        (*It)->replaceAllUsesWith((*It)->getOperand(0));
      }
      (*It)->eraseFromParent();
    }

    Changed = true; // Mark that changes were made
  } // End for (Function &F : M)


  


  // Print the final IR state before exiting


    // Perform a final pass to remove any no-op casts
  
  LLVM_DEBUG(dbgs() << "\n\n\n\n\nRemoving no-op casts:\n");
  for (Function &F : M) {
    SmallVector<Instruction *, 16> DeadCasts; // Collect casts to remove
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        // Check for ZExt, SExt, Trunc, or BitCast where source and dest types are identical
        if (auto *Cast = dyn_cast<CastInst>(&I)) {
          if (Cast->getSrcTy() == Cast->getDestTy()) {
             LLVM_DEBUG(dbgs() << "Found no-op cast: " << I << "\n");
             DeadCasts.push_back(&I);
          }
        }
      }
    }
    // Remove the collected casts after iterating through the block
    for (Instruction *DeadCast : DeadCasts) {
       LLVM_DEBUG(dbgs() << "Removing no-op cast: " << *DeadCast << "\n");
       DeadCast->replaceAllUsesWith(DeadCast->getOperand(0)); // Replace uses with the source operand
       DeadCast->eraseFromParent();
    }
  }

  // print the first and the last instruction that contains non-std types
  std::vector<Instruction *> NonStdTypes;
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (isNonStandardInt(I.getType())) {
          NonStdTypes.push_back(&I);
          break;
        }
      }
    }
  }

  if (NonStdTypes.size() > 0) {
    LLVM_DEBUG(dbgs() << "First non-std type instruction: " << *NonStdTypes[0] << "\n");
    LLVM_DEBUG(dbgs() << "Last non-std type instruction: " << *NonStdTypes[NonStdTypes.size() - 1] << "\n");
  }

  LLVM_DEBUG(dbgs() << "\n\n\n\n\nFinal module IR after HipPromoteIntsPass:\n");
  LLVM_DEBUG(M.print(dbgs(), nullptr));
  




  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}