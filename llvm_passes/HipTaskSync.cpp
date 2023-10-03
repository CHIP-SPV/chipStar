//===- HipTaskSync.cpp ---------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A pass to handle HIP cooperative group synchronization related operations.
//
//===----------------------------------------------------------------------===//

#include "HipTaskSync.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"



#include "llvm/Support/Debug.h"

#include <map>
#include <string>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <tuple>
#include <memory>

using namespace std;
using namespace llvm;

#define PASS_ID "hip-function-ptr"

namespace CooperativeGroups
{

    struct Sync
    {
        vector<Instruction *> insts;
        BasicBlock *block;
    };

    struct TiledPartition
    {
        vector<Instruction *> insts;
        BasicBlock *block;
        Value *size;
        Value *threadIdx;
    };

    struct CodeRegion
    {
        BasicBlock *startBB;
        BasicBlock *endBB;

        virtual ~CodeRegion() {} // Needed for polymorphism
        CodeRegion() = default;
        CodeRegion(BasicBlock *start, BasicBlock *end) : startBB(start), endBB(end) {}
    };

    struct Branch: CodeRegion
    {
        BasicBlock *entryBlock;
        BasicBlock *conditionBlock;
        CodeRegion thenBlocks;
        bool hasElse;
        CodeRegion elseBlocks;
        BasicBlock *mergeBlock;
        BasicBlock *exitBlock;

        Branch() = default;
        Branch(BasicBlock *entry, BasicBlock *condition,
           CodeRegion thenB, bool hasE, CodeRegion elseB, BasicBlock *merge, BasicBlock *exit)
        : CodeRegion(condition, merge), entryBlock(entry), conditionBlock(condition),
          thenBlocks(thenB), hasElse(hasE), elseBlocks(elseB), mergeBlock(merge), exitBlock(exit) {}
    };

    struct Loop: CodeRegion
    {
        BasicBlock *entryBlock;
        BasicBlock *preheaderBlock;
        BasicBlock *headerBlock;
        CodeRegion bodyBlocks;
        BasicBlock *latchBlock;
        BasicBlock *exitBlock;

        Loop() = default;
        Loop(BasicBlock *entry, BasicBlock *preheader, BasicBlock *header,
           CodeRegion body, BasicBlock *latch, BasicBlock *exit)
        : CodeRegion(preheader, header), entryBlock(entry), preheaderBlock(preheader), headerBlock(header),
          bodyBlocks(body), latchBlock(latch), exitBlock(exit) {}
    };

    struct LocalVariable
    {
        AddrSpaceCastInst *addrSpaceCast;
        vector<LoadInst*> references;
    };

    struct CGRegion: CodeRegion
    {
        Sync sync; // sync at the endBB

        CGRegion() = default;
        CGRegion(BasicBlock *start, BasicBlock *end, Sync sync)
        : CodeRegion(start, end), sync(sync) {}
    };

    struct SerializedCGRegion: CodeRegion
    {
        Value *threadIdx;

        SerializedCGRegion() = default;
        SerializedCGRegion(BasicBlock *start, BasicBlock *end, Value *threadIdx)
        : CodeRegion(start, end), threadIdx(threadIdx) {}
    };

    std::string formatInst(llvm::Instruction *instr)
    {
        if (!instr)
        {
            return "null";
        }
        std::string str;
        llvm::raw_string_ostream rso(str);

        instr->print(rso);

        std::string instrString = rso.str();
        if (instrString.length() > 100)
        {
            instrString = instrString.substr(0, 97) + "...";
        }

        return instrString;
    }

    std::string formatBB(llvm::BasicBlock *bb)
    {
        if (!bb)
        {
            return "null";
        }

        std::string output;
        llvm::raw_string_ostream os(output);

        os << "\n";
        for (auto &I : *bb)
        {
            os << "\t" << formatInst(&I) << "\n";
        }
        os << "\n";

        return os.str();
    }

    class KernelTranslator
    {
    public:
        KernelTranslator(Function &f) : function(f) {}

        bool run()
        {
            bool kernelModified = false, shouldStop = false;

            // if no tiled partition, the code doesn't use cg
            findTiledPartition();
            if (!hasTiledPartition())
            {
                dbgs() << "KernelTranslator: no tiled partition found, translation stops\n\n";
                return false;
            }

            // if no sync at all, no need to loop serialize
            analyzeKernel();
            if (syncs.size() == 0)
            {
                dbgs() << "KernelTranslator: no sync found, translation stops\n\n";
                return false;
            }

            // In theory, if there is no sync inside branch/loop,
            // we can enforce thread group sync to thread block sync
            // this won't change the program's behavior, so transformation
            // can be skipped. This is not implemented yet.
            // if (hasSyncInBranchOrLoop())
            // {
            //     dbgs() << "KernelTranslator: sync found in branch/loop, translation stops\n\n";
            //     return false;
            // }

            preprocessBasicBlocks();

            int i = 0;

            while (!shouldStop)
            {
                dbgs()<<"\n===============================================\n";
                analyzeKernel();

                {
                    Sync sync1, sync2;
                    if (hasAdjacentSyncs(sync1, sync2))
                    {
                        handleAdjacentSyncs(sync1, sync2);
                        kernelModified = true;
                        continue;
                    }
                }

                {
                    Branch ifElseBranch;
                    if (hasIfElse(ifElseBranch))
                    {
                        handleIfElse(ifElseBranch);
                        kernelModified = true;
                        continue;
                    }
                }
                {
                    Branch ifBranch;
                    Sync sync;
                    if (hasSyncInIf(ifBranch, sync))
                    {
                        handleSyncInIf(ifBranch, sync);
                        kernelModified = true;
                        continue;
                    }
                }
                {
                    Sync sync;
                    if (hasUnhandledSync(sync))
                    {
                        handleSync(sync);
                        kernelModified = true;
                        continue;
                    }
                }
                if (hasTiledPartition())
                {
                    BasicBlock *threadCheckBB;
                    createThreadCheckBeforeTiledPartition(threadCheckBB);
                    analyzeKernel();
                    transformLocalVariables();
                    analyzeKernel();
                    transformThreadIdxCalls();
                    removeTiledPartitionAndGroupSyncs(threadCheckBB);
                    kernelModified = true;
                    break;
                }
                shouldStop = true;
            }

            dbgs() << "KernelTranslator: kernel modified: " << kernelModified << "\n\n";

            return kernelModified;
        }

    private:
        int constexpr static ADDR_SPACE_SHARED = 3;
        int constexpr static ADDR_SPACE_CONSTANT = 4;
        int constexpr static ADDR_SPACE_LOCAL = 5;
        int constexpr static MAX_THREADS_PER_BLOCK = 512;

        Function &function;
        vector<Sync> syncs;
        TiledPartition tiledPartition;

        llvm::PostDominatorTree PDT;
        llvm::DominatorTree DT;
        llvm::LoopInfo LI;

        vector<BasicBlock *> basicBlocks;
        BasicBlock * kernelExitBlock;

        vector<shared_ptr<CodeRegion> > controlStructures;

        vector<LocalVariable> localVariables;
        vector<CallInst *> threadIdxCalls;
        std::vector<SerializedCGRegion> serializedCgRegions;

        struct GroupSyncHash {
            std::size_t operator()(const Sync& gs) const {
                return std::hash<llvm::Instruction*>()(gs.insts.front());
            }
        };
    
	struct GroupSyncEqual {
            bool operator()(const Sync& lhs, const Sync& rhs) const {
                return lhs.insts.front() == rhs.insts.front();
            }
        };
        std::unordered_set<Sync, GroupSyncHash, GroupSyncEqual> handledGroupSyncs;


        void preprocessBasicBlocks()
        {
            analyzeKernel();

            // add a sync to the end of the function
            // it is guaranteed to have at least one sync for cloning
            llvm::Function::iterator bb = function.end();
            --bb;  // go to the last basic block
            if (llvm::isa<llvm::ReturnInst>(bb->getTerminator())) {
                auto *syncInst = syncs.front().insts.front()->clone();
                syncInst->insertBefore(bb->getTerminator());
            }

            analyzeKernel();

            // cut the basic block in half by tiled partition instruction
            // tpEndInst should be in the predecessor block
            auto *tpEndInst = tiledPartition.insts.back();
            if (!tpEndInst->isTerminator())
            {
                llvm::BasicBlock *parentBlock = tpEndInst->getParent();
                llvm::BasicBlock::iterator iter(tpEndInst);
                ++iter;
                llvm::BasicBlock *newBlock = parentBlock->splitBasicBlock(iter);
            }
            auto *tpStartInst = tiledPartition.insts.front();
            if (&tpStartInst->getParent()->front() != tpStartInst)
            {
                llvm::BasicBlock *parentBlock = tpStartInst->getParent();
                llvm::BasicBlock::iterator iter(tpStartInst);
                llvm::BasicBlock *newBlock = parentBlock->splitBasicBlockBefore(iter);
            }

            analyzeKernel();

            // for every branch/loop, make sure conditional block(header block) has no irelevant instructions
            // this invalidates the records in controlStructures, analyzeKernel() is needed
            for (auto &cs : controlStructures)
            {
                // find start of dependent insts of the last inst of currBB
		auto *currBB = cs->startBB;
                auto *condInst = currBB->getTerminator();
                vector<Instruction *> dependentInsts;
                findRelevantInstructions(condInst, dependentInsts);
                auto startOfCondBlock = dependentInsts.back();
                // if the start of dependent insts is not the start of the currBB, perform partition
                if (startOfCondBlock != &currBB->front())
                {
                    dbgs()<<"preprocessBasicBlocks: start of control structure is not the start of currBB, perform partition\n";
                    dbgs()<<"\tActual start of BB"<<formatInst(&currBB->front())<<"\n";
                    dbgs()<<"\tExpected start of BB"<<formatInst(startOfCondBlock)<<"\n\n";
                    currBB->splitBasicBlockBefore(startOfCondBlock);
                }
            }

            analyzeKernel();

            // for every branch, make sure merge block has no irrelevant instructions
            // that means, the merge block has a terminator only
            // this invalidates the records in controlStructures, analyzeKernel() is needed
            for (auto &cs: controlStructures)
            {
                // if (auto branch = dynamic_pointer_cast<Branch>(cs))
                auto temp = cs;
		auto branch = (Branch *)temp.get();
		if (branch != nullptr)
		{
                    auto *mergeBlock = branch->mergeBlock;
                    // make sure that merge block contains terminator only
                    // if not, perform partition
                    if (mergeBlock->size() != 1) {
                        dbgs()<<"preprocessBasicBlocks: merge block is not empty, perform partition\n\n";
                        auto exitBlock = mergeBlock->splitBasicBlock(mergeBlock->begin());
                        // update info
                        branch->mergeBlock = mergeBlock;
                        branch->exitBlock = exitBlock;
                    }
                }
            }

            // make sure each sync is in a basic block by itself
            bool hasChanged = true;
            while (hasChanged)
            {
                analyzeKernel();
                hasChanged = false;
                for (auto &sync : syncs)
                {
                    if (!isSyncInBBOnly(sync))
                    {
                        separateOutSyncFromBB(sync);
			hasChanged = true;
                        break;
                    }
                }
            }

	    dbgs()<<"PreprocessBasicBlocks: done\n\n";
        }

        void analyzeKernel()
        {
            syncs.clear();
            controlStructures.clear();
            basicBlocks.clear();
            threadIdxCalls.clear();

            PDT.recalculate(function);
            DT.recalculate(function);
            LI.analyze(DT);

            findLocalVariables();

            unordered_set<BasicBlock *> visited;
            traverseCFG(tiledPartition.block, visited);

	    for (auto *currBB : basicBlocks)
            {
                for (auto &I : *currBB)
                {
                    // find cg sync statements
                    if (isSync(&I))
                    {
                        Sync groupSync;
                        groupSync.block = currBB;

                        vector<Instruction *> dependentInsts;
                        findRelevantInstructions(&I, dependentInsts);
                        groupSync.insts.insert(groupSync.insts.end(), dependentInsts.rbegin(), dependentInsts.rend());
                        // groupSync.insts.push_back(&I);

                        syncs.push_back(groupSync);
                    }
                    if (isThreadIdxCall(&I) && DT.dominates(tiledPartition.insts.back(), &I))
                    {
                        threadIdxCalls.push_back(dyn_cast<CallInst>(&I));
                    }
                }

                // find if/if-else branches
                Branch branch;
                if (isConditionBlockOfBranch(currBB, branch))
                {
                    controlStructures.push_back(make_shared<Branch>(branch));
                }
	    }

	    dbgs()<<"AnalyzeKernel: found "<<syncs.size()<<" group syncs and "<<threadIdxCalls.size()<<" threadIdx calls\n\n";

            // Sort control flow constructs by post dominance
            auto postDominanceComparator = [this](const shared_ptr<CodeRegion> &a, const shared_ptr<CodeRegion> &b) -> bool
            {
                return PDT.dominates(a->endBB, b->endBB);
            };
            std::sort(controlStructures.begin(), controlStructures.end(), postDominanceComparator);
        }

        void transformLocalVariables()
        {
            llvm::IRBuilder<> builder{function.getContext()};

            llvm::Value *arraySize = tiledPartition.size;

            for (auto &localVar : localVariables)
            {
                // Create a global shared array for each local variable
                builder.SetInsertPoint(localVar.addrSpaceCast->getNextNode());
                llvm::Type *elementType = localVar.addrSpaceCast->getType()->getPointerElementType();
                llvm::Type *arrayType = llvm::ArrayType::get(elementType, MAX_THREADS_PER_BLOCK);
                llvm::GlobalVariable *globalArray = new llvm::GlobalVariable(
                    *function.getParent(),
                    arrayType,
                    false,
                    llvm::GlobalValue::InternalLinkage,
                    llvm::ConstantAggregateZero::get(arrayType),
                    localVar.addrSpaceCast->getName().str() + "_array",
                    nullptr,
                    llvm::GlobalVariable::NotThreadLocal,
                    ADDR_SPACE_SHARED
                );
                globalArray->setAlignment(llvm::Align(4));

                // Copy value to shared array before tiled partition and thread check starts
                auto serializeVariablesBB = tiledPartition.block->getSinglePredecessor()->getSinglePredecessor();
                builder.SetInsertPoint(serializeVariablesBB->getTerminator());
                auto index = tiledPartition.threadIdx;
                llvm::Type *castedArrayType = llvm::PointerType::get(arrayType, ADDR_SPACE_CONSTANT);
                auto castedArrayAlloca = builder.CreateAddrSpaceCast(globalArray, castedArrayType, "castedArrayAlloca");
                auto gep = builder.CreateGEP(arrayType, castedArrayAlloca, {builder.getInt64(0), index});
                auto localVarInGenericAddrSpace = builder.CreateLoad(localVar.addrSpaceCast->getType()->getPointerElementType(), localVar.addrSpaceCast, "localvarcopy");
		builder.CreateStore(localVarInGenericAddrSpace, gep);

                // Replace all references to local variable with references to shared array
                for (auto *loadInst : localVar.references)
                {
                    // Find the cgregion that the loadInst belongs to
                    SerializedCGRegion cgRegion;
                    bool regionFound = false;
                    for (auto &region : serializedCgRegions)
                    {
                        if (isBasicBlockWithinRegion(region, loadInst->getParent())) {
                            cgRegion = region;
                            regionFound = true;
                            break;
                        }
                    }

                    if (!regionFound)
                    {
                        dbgs()<<"TransformLocalVariables: cgRegion not found\n\n";
                        continue;
                    }

                    // Build threadIdx
                    builder.SetInsertPoint(loadInst);
                    auto loopIndex = cgRegion.threadIdx;
                    if (!cgRegion.threadIdx)
                    {
                        dbgs()<<"TransformLocalVariables: cgRegion.threadIdx not found\n\n";
                        dbgs()<<"startBB: "<<formatBB(cgRegion.startBB);
                        dbgs()<<"endBB: "<<formatBB(cgRegion.endBB);
                        continue;
                    }
                    loopIndex = builder.CreateZExt(loopIndex, builder.getInt64Ty()); // Cast from i32 to i64
                    auto loopSerializeThreadIndex = tiledPartition.threadIdx;
                    if (!tiledPartition.threadIdx)
                    {
                        dbgs()<<"TransformLocalVariables: tiledPartition.threadIdx not found\n\n";
                        continue;
                    }
                    auto index = builder.CreateAdd(loopIndex, loopSerializeThreadIndex); // Emulated thread index = threadIndex + loopIndex
		    
		    // Replace the loadInst with a load from the shared array
                    auto gep = builder.CreateGEP(arrayType, globalArray, { builder.getInt32(0), index } );
                    loadInst->setOperand(0, gep);
                    // loadInst->replaceAllUsesWith(index);
                }
            }

            dbgs()<<"TransformLocalVariables: done\n\n";
        }

        void transformThreadIdxCalls()
        {
            queue<Instruction*> threadIdxUser;
            for (auto callInst : threadIdxCalls)
            {
                bool regionFound = false;
                SerializedCGRegion cgRegion;
                for (auto &region : serializedCgRegions)
                {
                    // dbgs()<<"Region.start"<<formatBB(region.startBB);
                    // dbgs()<<"Region.end"<<formatBB(region.endBB);
                    // dbgs()<<"CallInst.parent"<<formatBB(callInst->getParent());
                    if (isBasicBlockWithinRegion(region, callInst->getParent())) {
                        cgRegion = region;
                        regionFound = true;
                        break;
                    }
                }

                if (!regionFound)
                {
                    dbgs()<<"TransformThreadIdxCalls: cgRegion not found\n\n";
                    continue;
                }

                dbgs()<<"Cgregion.threadIdx: ";
                cgRegion.threadIdx->print(dbgs());
                dbgs()<<"\n";

                // replace all uses with cgRegion.threadIdx, except for trunc
                // If the use is trunc, replace its uses with cgRegion.threadIdx
                vector<Value *> threadIdxUsers(callInst->user_begin(), callInst->user_end());
                for (auto &user : threadIdxUsers)
                {
                    auto userInst = dyn_cast<Instruction>(user);
                    if (!userInst)
                    {
                        continue;
			}

                    // Find the cgregion that the user belongs to
                    SerializedCGRegion cgRegion;
                    bool regionFound = false;
                    for (auto &region : serializedCgRegions)
                    {
                        if (isBasicBlockWithinRegion(region, userInst->getParent())) {
                            cgRegion = region;
                            regionFound = true;
                            break;
                        }
                    }
                    if (!regionFound)
                    {
                        dbgs()<<"TransformThreadIdxCalls: cgRegion not found\n\n";
                        // print userInst
                        dbgs()<<"userInst: "<<formatInst(userInst)<<"\n";
                        continue;
                    }

                    // Build threadIdx
                    IRBuilder<> builder{function.getContext()};
                    builder.SetInsertPoint(userInst);
                    auto loopIndex = cgRegion.threadIdx;
                    auto loopSerializeThreadIndex = tiledPartition.threadIdx;
                    if (userInst->getType()->isIntegerTy(64))
                    {
                        loopIndex = builder.CreateZExt(loopIndex, builder.getInt64Ty()); // Cast from i32 to i64
                    }
                    else
                    {
                        loopSerializeThreadIndex = builder.CreateTrunc(loopSerializeThreadIndex, builder.getInt32Ty()); // Cast from i64 to i32
                    }

                    if (!tiledPartition.threadIdx)
                    {
                        dbgs()<<"TransformThreadIdxCalls: tiledPartition.threadIdx not found\n\n";
                        continue;
                    }
                    auto index = builder.CreateAdd(loopIndex, loopSerializeThreadIndex); // Emulated thread index = threadIndex + loopIndex

                    // print out user
                    dbgs()<<"TransformThreadIdxCalls: user: "<<formatInst(dyn_cast<Instruction>(user))<<"\n";
                    if (isa<TruncInst>(userInst))
                    {
                        userInst->replaceAllUsesWith(index);
                        userInst->eraseFromParent();
			}
                    else
                    {
                        userInst->replaceUsesOfWith(callInst, index);
                    }
                }


                // remove the callInst
                callInst->eraseFromParent();
            }
        }

        void removeTiledPartitionAndGroupSyncs(BasicBlock *threadCheckBB)
        {
            // By now, all syncs and tiled partition should exist
            // in basic blocks by itself

            clearBasicBlock(tiledPartition.block);

            // Make sure all threads must go through thread block syncs
            // before exiting the kernel. To achieve this, we will redirect
            // thread check to jump across all thread block syncs, then
            // stop at the kernel exit block
            {
                std::vector<BasicBlock*> threadBlockSyncBBs;
                for (auto& sync : syncs) {
                    if (isThreadBlockSync(sync.insts.front())) {
                        threadBlockSyncBBs.push_back(sync.block);
                    }
                }
                dbgs()<<"Found "<<threadBlockSyncBBs.size()<<" thread block syncs\n";

                if (threadBlockSyncBBs.size() > 0)
                {
                    auto prevBB = threadCheckBB;
                    dbgs()<<"threadCheckBB: "<<formatBB(threadCheckBB);

                    for (auto &threadBlockSyncBB : threadBlockSyncBBs)
                    {
                        prevBB->getTerminator()->setSuccessor(1, threadBlockSyncBB);
                        if (threadBlockSyncBB != kernelExitBlock)
                        {
                            auto terminatorClone = prevBB->getTerminator()->clone();
                            terminatorClone->setSuccessor(0, threadBlockSyncBB->getTerminator()->getSuccessor(0));
                            ReplaceInstWithInst(threadBlockSyncBB->getTerminator(), terminatorClone);
                            prevBB = threadBlockSyncBB;
                        }
			}

                    if (threadBlockSyncBBs.back() != kernelExitBlock)
                    {
                        threadBlockSyncBBs.back()->getTerminator()->setSuccessor(1, kernelExitBlock);
                    }
                }
            }


            // Remove group syncs
            for (auto &sync : syncs)
            {
                if (isThreadGroupSync(sync.insts.front()))
                {
                    clearBasicBlock(sync.block);
                }
            }
        }

        bool hasAdjacentSyncs(Sync &sync1, Sync &sync2)
        {
            if (syncs.size() < 2)
            {
                return false;
            }

            for (auto it = syncs.begin(); it != syncs.end() - 1; ++it)
            {
                if (it->block->getSingleSuccessor() == (it + 1)->block)
                {
                    sync1 = *it;
                    sync2 = *(it + 1);
                    return true;
                }
            }

	    return false;
        }

        void handleAdjacentSyncs(Sync sync1, Sync sync2)
        {
            auto sync1Terminator = sync1.block->getTerminator();
            auto sync2Terminator = sync2.block->getTerminator();

            auto sync2TerminatorClone = sync2Terminator->clone();

            ReplaceInstWithInst(sync1Terminator, sync2TerminatorClone);

            sync2.block->eraseFromParent();
            dbgs() << "HandleAdjacentSyncs: transformation applied\n\n";
        }

        bool hasIfElse(Branch &ifElseBranch)
        {
            for (auto it = controlStructures.rbegin(); it != controlStructures.rend(); ++it)
            {
                // if (auto branch = dynamic_pointer_cast<Branch>(*it))
                auto temp = it;
                auto branch = (Branch* )temp->get();
		{
                    if (!branch->hasElse)
                    {
                        continue;
                    }
                    ifElseBranch = *branch;
                    return true;
                }
            }
            return false;
        }

        void handleIfElse(Branch ifElseBranch)
        {
            // point condBB to thenBB and mergeBB
            {
                auto *branchInst = dyn_cast<BranchInst>(ifElseBranch.conditionBlock->getTerminator());
                branchInst->setSuccessor(0, ifElseBranch.thenBlocks.startBB);
                branchInst->setSuccessor(1, ifElseBranch.mergeBlock);
            }

            // duplicate condBB and mergeBB
            ValueToValueMapTy VMap;
            auto *cond2BB = CloneBasicBlock(ifElseBranch.conditionBlock, VMap, ".cond2BB", &function);
            for (auto &inst : *cond2BB)
            {
		for (auto &U : inst.operands())
                {
                    Value *usedValue = U.get();
                    if (Value *clonedValue = VMap[usedValue])
                    {
                        U.set(clonedValue);
                    }
                }
            }

            auto *merge2BB = CloneBasicBlock(ifElseBranch.mergeBlock, VMap, ".merge2BB", &function);
            for (auto &inst : *merge2BB)
            {
                for (auto &U : inst.operands())
                {
                    Value *usedValue = U.get();
                    if (Value *clonedValue = VMap[usedValue])
                    {
                        U.set(clonedValue);
                    }
                }
            }

            // point mergeBB to cond2BB
            {
                auto *branchInst = dyn_cast<BranchInst>(ifElseBranch.mergeBlock->getTerminator());
                branchInst->setSuccessor(0, cond2BB);
            }

            // point cond2BB to merge2BB and elseBB
            // sucessor order is switched to create "not cond" effect
            {
                auto *branchInst = dyn_cast<BranchInst>(cond2BB->getTerminator());
                branchInst->setSuccessor(1, ifElseBranch.elseBlocks.startBB);
                branchInst->setSuccessor(0, merge2BB);
            }

            // point elseBB to merge2BB
            {
                auto *branchInst = dyn_cast<BranchInst>(ifElseBranch.elseBlocks.endBB->getTerminator());
                branchInst->setSuccessor(0, merge2BB);
            }

	     // point merge2BB to exitBB
            {
                auto *branchInst = dyn_cast<BranchInst>(merge2BB->getTerminator());
                branchInst->setSuccessor(0, ifElseBranch.exitBlock);
            }

            dbgs() << "HandleIfElse: transformation applied\n\n";
        }

        bool hasSyncInIf(Branch &ifBranch, Sync &sync)
        {
            for (auto it = controlStructures.rbegin(); it != controlStructures.rend(); ++it)
            {
                // auto branch = dynamic_pointer_cast<Branch>(*it);
		// This is not an elegance process, not for no RTTI support
		auto temp = it;
		auto branch = (Branch* )temp->get();

                if (!branch)
                {
                    continue;
                }

                if (branch->hasElse)
                {
                    continue;
                }


                for (auto &groupSync : syncs)
                {
                    // Check that sync is in the middle of branch
                    if (isBasicBlockWithinRegion(*branch, groupSync.block))
                    {
                        sync = groupSync;
                        ifBranch = *branch;
                        dbgs() << "hasSyncInIf: sync found in if\n\n";
                        return true;
		    }
		}
	    }

	    return false;
        }

        void handleSyncInIf(Branch branch, Sync sync)
        {
            if (!isBasicBlockWithinRegion(branch, sync.block))
            {
                dbgs() << "HandleSyncInIf: sync is not in then block\n\n";
                return;
            }

            auto result = tuple<BasicBlock *, BasicBlock *, BasicBlock *>{sync.block->getSinglePredecessor(), sync.block, sync.block->getSingleSuccessor()};

            // CFG is changed
            DT.recalculate(function);

	    auto then1 = CodeRegion{findStartOfRegion(get<0>(result), branch.conditionBlock), get<0>(result)};
            auto *syncBB = get<1>(result);
            auto then2 = CodeRegion{get<2>(result), findEndOfRegion(get<2>(result), branch.mergeBlock)};

            // output separated results
            dbgs()<<"HandleSyncInIf: then1: "<<formatBB(then1.startBB)<<"\n";
            dbgs() << "HandleSyncInIf: before: " << formatBB(get<0>(result)) << "\n";
            dbgs() << "HandleSyncInIf: middle: " << formatBB(get<1>(result)) << "\n";
            dbgs() << "HandleSyncInIf: after: " << formatBB(get<2>(result)) << "\n\n";
            dbgs()<<"HandleSyncInIf: then2: "<<formatBB(then2.endBB)<<"\n\n";

            dbgs()<<"HandleSyncInIf: branch: "<<formatBB(branch.conditionBlock)<<"\n";
            dbgs()<<"HandleSyncInIf: merge: "<<formatBB(branch.mergeBlock)<<"\n";

            dbgs() << "HandleSyncInIf: transformation applied in " << formatInst(branch.conditionBlock->getTerminator()) << "\n\n";

            // point then1 to mergeBB 
            {
                auto *branchInst = dyn_cast<BranchInst>(then1.endBB->getTerminator());
                branchInst->setSuccessor(0, branch.mergeBlock);
            }


            // point mergeBB to syncBB
            {
                auto *branchInst = dyn_cast<BranchInst>(branch.mergeBlock->getTerminator());
                branchInst->setSuccessor(0, syncBB);
            }

            // duplicate syncBB and mergeBB and place them in the function
            // replace usage of old instructions with newly cloned instructions
            ValueToValueMapTy VMap;
            auto *cond2BB = CloneBasicBlock(branch.conditionBlock, VMap, ".cond2BB", &function);
            for (auto &inst : *cond2BB)
            {
                for (auto &U : inst.operands())
                {
                    Value *usedValue = U.get();
                    if (Value *clonedValue = VMap[usedValue])
                    {
                        U.set(clonedValue);
                    }
                }
            }
            auto *merge2BB = CloneBasicBlock(branch.mergeBlock, VMap, ".merge2BB", &function);
            for (auto &inst : *merge2BB)
            {
                for (auto &U : inst.operands())
		{
                    Value *usedValue = U.get();
                    if (Value *clonedValue = VMap[usedValue])
                    {
                        U.set(clonedValue);
                    }
                }
            }

            // this replaces usage of old instructions with newly cloned instructions

            // point syncBB to cond2BB
            {
                auto *branchInst = dyn_cast<BranchInst>(syncBB->getTerminator());
                branchInst->setSuccessor(0, cond2BB);
            }

            // point cond2BB to then2BB and merge2BB
            {
                auto *branchInst = dyn_cast<BranchInst>(cond2BB->getTerminator());
                branchInst->setSuccessor(0, then2.startBB);
                branchInst->setSuccessor(1, merge2BB);
            }

            // point then2BB to merge2BB
            {
                auto *branchInst = dyn_cast<BranchInst>(then2.endBB->getTerminator());
                branchInst->setSuccessor(0, merge2BB);
            }

            // point merge2BB to exitBB
            {
                auto *branchInst = dyn_cast<BranchInst>(merge2BB->getTerminator());
                branchInst->setSuccessor(0, branch.exitBlock);
            }
        }

        bool hasUnhandledSync(Sync &sync)
        {
            for (auto &gs: syncs)
            {
                if (handledGroupSyncs.find(gs) == handledGroupSyncs.end())
                {
                    sync = gs;
                    return true;
                }
            }

	    return false;
        }

        void handleSync(Sync sync)
        {
            auto cgRegion = extractCGRegion(sync);

            auto cgLoop = createStructureForCGRegion(cgRegion);

            auto serializedCgRegion = serializeCGRegion(cgLoop, tiledPartition.size);

            serializedCgRegions.push_back(serializedCgRegion);

            handledGroupSyncs.insert(sync);

            dbgs() << "HandleSync: transformation applied\n\n";
        }

        void findLocalVariables()
        {
            localVariables.clear();

            auto entryBlock = &function.getEntryBlock();

            for (auto &I : *entryBlock)
            {
                auto addrCastInst = dyn_cast<AddrSpaceCastInst>(&I);
                if (!addrCastInst)
                {
                    continue;
                }

                dbgs()<<"FindLocalVariables: addrCastInst found: "<<formatInst(addrCastInst)<<"\n";

                auto addrSpace = addrCastInst->getDestAddressSpace();
                if (addrSpace != ADDR_SPACE_CONSTANT && addrSpace != ADDR_SPACE_LOCAL)
                {
                    continue;
                }

                dbgs()<<"FindLocalVariables: addrCastInst is of correct address space\n";

                auto allocainst = dyn_cast<AllocaInst>(addrCastInst->getOperand(0));
                if (!allocainst || allocainst->getParent() != entryBlock)
                {
                    continue;
                }

                dbgs()<<"FindLocalVariables: allocaInst found: "<<formatInst(allocainst)<<"\n";

                // Check if the alloca is NOT of a struct/class type
		Type* allocatedType = allocainst->getAllocatedType();
                if (allocatedType->isStructTy())  // Check if it's a struct type
                {
                    continue;
                }

                dbgs()<<"FindLocalVariables: allocaInst is of correct type\n";


                vector<LoadInst *> references;
                for (auto *user : addrCastInst->users())
                {
                    dbgs()<<"FindLocalVariables: user: "<<formatInst(dyn_cast<Instruction>(user))<<"\n";
                    auto loadInst = dyn_cast<LoadInst>(user);
                    if (!loadInst)
                    {
                        continue;
                    }

                    dbgs()<<"FindLocalVariables: user: loadInst found: "<<formatInst(loadInst)<<"\n";

                    if (!DT.dominates(tiledPartition.insts.back(), loadInst))
                    {
                        continue;
                    }

                    dbgs()<<"FindLocalVariables: user: loadInst is after tiled partition\n";

                    references.push_back(dyn_cast<LoadInst>(user));
                }

                if (references.size() == 0)
                {
                    continue;
                }

                dbgs()<<"FindLocalVariables: "<<references.size()<<" references found\n";

                // Note that addrCastInst returns a pointer to the variable
                localVariables.push_back(LocalVariable{addrCastInst, references});
            }

            dbgs()<<"FindLocalVariables: done, collected "<<localVariables.size()<<" local variables\n\n";
        }

        void traverseCFG(BasicBlock *currBB, unordered_set<BasicBlock *> &visited)
        {
            if (!currBB || visited.find(currBB) != visited.end())
            {
		return;
            }

            visited.insert(currBB);
            basicBlocks.push_back(currBB);

            // dbgs() << "TraverseCFG: current BB terminator: " << formatInst(currBB->getTerminator()) << "\n\n";

            if (currBB->getTerminator()->getNumSuccessors() == 0)
            {
                kernelExitBlock = currBB;
                return;
            }

            for (auto succBB : successors(currBB))
            {
                traverseCFG(succBB, visited);
            }
        }

        bool hasTiledPartition()
        {
            return tiledPartition.block && tiledPartition.size;
        }

        void createThreadCheckBeforeTiledPartition(BasicBlock *&threadCheckBB)
        {
            auto threadIdxBB = tiledPartition.block->splitBasicBlockBefore(tiledPartition.block->begin(), "serializeVariables");
            threadCheckBB = tiledPartition.block->splitBasicBlockBefore(tiledPartition.block->begin(), "threadCheck");
            auto groupNumVal = tiledPartition.size;

            IRBuilder<> Builder(function.getContext());
            Builder.SetInsertPoint(threadIdxBB->getFirstNonPHI());

            // Find the function for getting local ID in the module
            Function *getLocalIDFunc = tiledPartition.block->getModule()->getFunction("_Z12get_local_idj");
            if (!getLocalIDFunc)
            {
                // Handle error: function not found
                dbgs() << "createThreadCheckBeforeTiledPartition: sync function(_Z12get_local_idj) not found\n\n";
                return;
            }

            // Create the argument: an i32 with the value 0
            Value *arg = Builder.getInt32(0);

            // Build call to _Z12get_local_idj to get thread ID
            CallInst *threadIDCall = Builder.CreateCall(getLocalIDFunc, {arg});

            threadIDCall->setCallingConv(CallingConv::SPIR_FUNC);
            threadIDCall->addAttributeAtIndex(1, Attribute::NoUndef);
            threadIDCall->addFnAttr(Attribute::Convergent);
	    threadIDCall->addFnAttr(Attribute::NoUnwind);


            Builder.SetInsertPoint(threadCheckBB->getFirstNonPHI());

            // Make sure that groupNumVal is of the same type as threadIDCall
            Value *groupNumValExtended = Builder.CreateSExt(groupNumVal, threadIDCall->getType());
            tiledPartition.threadIdx = threadIDCall;

            // Calculate the remainder
            Value *modVal = Builder.CreateSRem(threadIDCall, groupNumValExtended);

            // Create the comparison with 0 of the correct type
            Value *zero = ConstantInt::get(threadIDCall->getType(), 0);
            Value *condVal = Builder.CreateICmpEQ(modVal, zero, "TIDCmp");


            // Continue to successor only if thread id is 0
            Builder.CreateCondBr(condVal, threadCheckBB->getSingleSuccessor(), kernelExitBlock);

            // output sucessorBB and kernelExitBlock
            dbgs() << "AddThreadCheckBeforeTiledPartition: sucessorBB: " << formatBB(threadCheckBB->getSingleSuccessor()) << "\n";
            dbgs() << "AddThreadCheckBeforeTiledPartition: kernelExitBlock: " << formatBB(kernelExitBlock) << "\n\n";

            // Remove the terminator instruction from the original BB
            threadCheckBB->getTerminator()->eraseFromParent();

            dbgs() << "AddThreadCheckBeforeTiledPartition: transformation applied\n\n";
        }

        bool isConditionBlockOfBranch(BasicBlock *curr, Branch &branch)
        {
            if (!dyn_cast<BranchInst>(curr->getTerminator()))
            {
                return false;
            }

            if (curr->getTerminator()->getNumSuccessors() != 2)
            {
                return false;
            }

            auto *thenBB = curr->getTerminator()->getSuccessor(0);
	    auto *elseBB = curr->getTerminator()->getSuccessor(1);

            if (!thenBB || !elseBB)
            {
                return false;
            }

            // branch must not contain backedge
            if (LI.getLoopFor(thenBB) || LI.getLoopFor(elseBB))
            {
                dbgs() << "isConditionBlockOfBranch: branch contains backedge\n\n";
                return false;
            }

            BasicBlock *mergeBB = nullptr;

            // mergeBB must be the post-dominator of thenBB and elseBB
            mergeBB = PDT.findNearestCommonDominator(thenBB, elseBB);

            if (!mergeBB)
            {
                dbgs() << "isConditionBlockOfBranch: mergeBB is null\n\n";
                return false;
            }

            // we reinforce that only elseBB may be empty
            if (mergeBB == thenBB)
            {
                swap(thenBB, elseBB);
            }

            dbgs() << "isConditionBlockOfBranch: \n\tcondBB: \"" << formatBB(curr) << "\"\n";
            if (mergeBB != elseBB)
            {
                dbgs() << "\tthenBB: " << formatBB(thenBB) << "\n";
                dbgs() << "\telseBB: " << formatBB(elseBB) << "\n";
            }
            else
            {
                dbgs() << "\tthenBB: " << formatBB(thenBB) << "\n";
            }
            dbgs() << "\tmergeBB: " << formatBB(mergeBB) << "\n\n";
            dbgs() << "\n";

            branch.entryBlock = curr->getSinglePredecessor();
            branch.conditionBlock = curr;
            branch.thenBlocks = CodeRegion{thenBB, findEndOfRegion(thenBB, mergeBB)};
            branch.hasElse = (mergeBB != elseBB);
            if (branch.hasElse)
	    {
                branch.elseBlocks = CodeRegion{elseBB, findEndOfRegion(elseBB, mergeBB)};
            }
            else
            {
                branch.elseBlocks = CodeRegion{nullptr, nullptr};
            }
            branch.mergeBlock = mergeBB;
            branch.exitBlock = mergeBB->getSingleSuccessor();

            branch.startBB = branch.conditionBlock;
            branch.endBB = branch.mergeBlock;



            return true;
        }

        // Find CGregion that contains the given sync as the end of the region
        // This assumes the start BB is the dominator of the end BB
        CGRegion extractCGRegion(Sync sync)
        {
            BasicBlock *matchedBB = nullptr;
            std::queue<BasicBlock *> predecessorBBs;
            predecessorBBs.push(sync.block);

            while (!predecessorBBs.empty())
            {
                auto *currBB = predecessorBBs.front();
                predecessorBBs.pop();

                for (auto *predBB : predecessors(currBB))
                {
                    predecessorBBs.push(predBB);
                }

                if (currBB == sync.block)
                {
                    continue;
                }

                // Match currBB in the list of groupsyncs
                for (auto &groupSync : syncs)
                {
                    // Nearest BB with sync has found
                    if (groupSync.block == currBB)
                    {
                        matchedBB = currBB;
                    }
                }

		// Tiled partition can also be treated like a sync statement
                if (currBB == tiledPartition.block)
                {
                    matchedBB = currBB;
                }

                if (matchedBB)
                {
                    break;
                }
            }

            // MatchedBB is right before the starting BB of the CGRegion
            auto startBB = matchedBB->getSingleSuccessor();

            // This should not happen
            // In preprocessBasicblocks, we have made sure conditional block
            // should not contain irrelevant instructions like sync
            // so currBB cannot have conditional branch
            if (!startBB)
            {
                dbgs() << "ExtractCGRegion: startBB is null\n\n";
                return {nullptr, nullptr, sync};
            }


            // auto result = separateOutSyncFromBB(sync);
            auto endBB = sync.block->getSinglePredecessor();

            dbgs()<<"ExtractCGRegion: startBB: "<<formatBB(startBB)<<"\n";
            dbgs()<<"ExtractCGRegion: endBB: "<<formatBB(endBB)<<"\n\n";

            return CGRegion{startBB, endBB, sync};
        }

        Loop createStructureForCGRegion(CGRegion region)
        {
            Loop cgLoop;

            auto id = std::to_string(handledGroupSyncs.size());

            cgLoop.preheaderBlock = region.startBB->splitBasicBlockBefore(region.startBB->begin(), "cgLoopPreheader"+id);

            cgLoop.headerBlock = region.startBB->splitBasicBlockBefore(region.startBB->begin(), "cgLoopHeader"+id);

            cgLoop.latchBlock = region.endBB->splitBasicBlock(region.endBB->getTerminator(), "cgLoopLatch"+id);

            cgLoop.bodyBlocks = CodeRegion{ region.startBB, findEndOfRegion(region.startBB, cgLoop.latchBlock) };

            cgLoop.exitBlock = cgLoop.latchBlock->getSingleSuccessor();
	    cgLoop.exitBlock->setName("cgLoopExit"+id);
            if(!cgLoop.exitBlock)
            {
                dbgs() << "CreateStructureForCGRegion: exiting block is null\n\n";
            }

            cgLoop.latchBlock->getTerminator()->setSuccessor(0, cgLoop.headerBlock);

            return cgLoop;
        }

        SerializedCGRegion serializeCGRegion(Loop cgLoop, Value *groupSize)
        {
            IRBuilder<> Builder(cgLoop.preheaderBlock->getContext());

            // Create iterator
            Builder.SetInsertPoint(cgLoop.preheaderBlock->getFirstNonPHI());
            Value *allocIter = Builder.CreateAlloca(Type::getInt32Ty(cgLoop.preheaderBlock->getContext())); // i
            Value *const0 = Builder.getInt32(0);
            Builder.CreateStore(const0, allocIter); // i = 0
            LoadInst *IterLoadInst = Builder.CreateLoad(Type::getInt32Ty(cgLoop.preheaderBlock->getContext()), allocIter);


            // Create prologue
            Builder.SetInsertPoint(cgLoop.headerBlock->getFirstNonPHI());
            PHINode *PhiVal = Builder.CreatePHI(IterLoadInst->getType(), 2);
            PhiVal->addIncoming(IterLoadInst, cgLoop.preheaderBlock);
            Value *condVal = Builder.CreateICmpUGE(PhiVal, tiledPartition.size); // i >= GroupNum
            Builder.CreateCondBr(condVal, cgLoop.exitBlock, cgLoop.bodyBlocks.startBB);// if (i >= GroupNum ) then goto SuccBB else goto LoopHeaderBB
            cgLoop.headerBlock->getTerminator()->eraseFromParent();

            // Create epilogue
            Builder.SetInsertPoint(cgLoop.latchBlock->getFirstNonPHI());
            Value *IterVal = Builder.CreateAdd(PhiVal, Builder.getInt32(1)); // i = i + 1
            // Setup phi node related predecessor
            PhiVal->addIncoming(IterVal, cgLoop.latchBlock);

	    return SerializedCGRegion{cgLoop.preheaderBlock, cgLoop.exitBlock, PhiVal};
        }

        void findTiledPartition()
        {
            tiledPartition.block = nullptr;
            tiledPartition.size = nullptr;
            tiledPartition.insts.clear();

            for (auto &BB : function)
            {
                for (auto &I : BB)
                {
                    if (isTiledPartition(&I))
                    {
                        tiledPartition.block = &BB;
                        tiledPartition.size = dyn_cast<CallInst>(&I)->getArgOperand(2);

                        vector<Instruction *> dependentInsts;
                        findRelevantInstructions(&I, dependentInsts);
                        tiledPartition.insts.insert(tiledPartition.insts.end(), dependentInsts.rbegin(), dependentInsts.rend());
                        // tiledPartition.insts.push_back(&I);

                        // print out dependent instructions
                        dbgs() << "FindTiledPartition: dependent instructions: \n";
                        for (auto inst : tiledPartition.insts)
                        {
                            dbgs() << "\t" << formatInst(inst) << "\n";
                        }
                        dbgs() << "\n";

                        return;
                    }
                }
            }
        }

        // NEED FIX
        void findRelevantInstructions(Instruction *sync, vector<Instruction *> &dependentInsts)
        {
            unordered_set<Instruction *> instVisited;
            queue<Instruction *> instQueue;

	    instQueue.push(sync);
            dependentInsts.push_back(sync);
            Instruction *startOfSync = sync;

            while (!instQueue.empty())
            {
                Instruction *inst = instQueue.front();
                instQueue.pop();

                for (auto iter = inst->operands().begin(); iter != inst->operands().end(); ++iter)
                {
                    auto *newInst = dyn_cast<Instruction>(iter->get());

                    // Assume all alloc instructions are placed at the beginning of function
                    // We skip alloca instructions because they should always be at the top
                    if (!newInst || isa<AllocaInst>(newInst) || instVisited.find(newInst) != instVisited.end())
                    {
                        continue;
                    }

                    // Detect callInst that is immediately before addrspacecastInst
                    if (isa<AddrSpaceCastInst>(inst) && isa<CallInst>(inst->getPrevNode()))
                    {
                        if (newInst == inst->getPrevNode()->getOperand(0))
                        {
                            if (inst->getPrevNode()->comesBefore(startOfSync))
                            {
                                dependentInsts.push_back(inst->getPrevNode());
                                startOfSync = inst->getPrevNode();
                            }
                            instQueue.push(inst->getPrevNode());
                            instVisited.insert(inst->getPrevNode());
                        }
                        continue;
                    }

                    if (newInst->getNumUses() > 1)
                    {
                        continue;
                    }

                    if (newInst->comesBefore(startOfSync))
                    {
                        startOfSync = newInst;
                    }

		    dependentInsts.push_back(newInst);
                    instQueue.push(newInst);
                    instVisited.insert(newInst);
                }
            }
        }

        CallInst *findCallInst(std::string name)
        {
            for (llvm::BasicBlock &BB : function)
            {
                for (llvm::Instruction &I : BB)
                {
                    if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&I))
                    {
                        if (auto *CalledFunc = Call->getCalledFunction())
                        {
                            if (CalledFunc->getName() == name)
                            {
                                return Call;
                            }
                        }
                    }
                }
            }
            return nullptr;
        }

        bool isSyncInBBOnly(Sync sync)
        {
            auto currBB = sync.block;
            auto startOfSync = sync.insts.back()->getIterator();
            auto endOfSync = sync.insts.front()->getIterator();

	    return (startOfSync == currBB->begin()
            && std::distance(pred_begin(currBB), pred_end(currBB)) <= 1
            && currBB->getUniquePredecessor()->getTerminator()->getNumSuccessors() <= 1
            && endOfSync == prev(currBB->end(), 2)
            && currBB->getTerminator()->getNumSuccessors() <= 1
            && std::distance(pred_begin(currBB), pred_end(currBB)) <= 1);
        }

        tuple<BasicBlock *, BasicBlock *, BasicBlock *> separateOutSyncFromBB(Sync sync)
        {
            dbgs() << "separateOutSyncFromBB: sync BB before split" << formatBB(sync.block) << "\n\n";

            auto currBB = sync.block;
            auto startOfSync = sync.insts.back()->getIterator();
            auto endOfSync = sync.insts.front()->getIterator();


            // dbgs()<<"start of sync"<<formatInst(&*startOfSync)<<"\n";
            // dbgs()<<"conditions"<<(startOfSync != currBB->begin())<<std::distance(pred_begin(currBB), pred_end(currBB))<<currBB->getUniquePredecessor()->getTerminator()->getNumSuccessors()<<"\n\n";
            if (
                startOfSync != currBB->begin()
            || std::distance(pred_begin(currBB), pred_end(currBB)) > 1
            || currBB->getUniquePredecessor()->getTerminator()->getNumSuccessors() > 1
            )
            {
                currBB->splitBasicBlockBefore(startOfSync);
            }

            dbgs()<<"1\n";
            dbgs()<<formatBB(currBB->getSinglePredecessor())<<"\n\n";
            dbgs()<<formatBB(currBB)<<"\n\n";

            // dbgs()<<"back split?"<<(endOfSync != prev(currBB->end()))<<(currBB->getTerminator()->getNumSuccessors() > 1)<<"\n\n";
            if (
                endOfSync != prev(currBB->end())
                || currBB->getTerminator()->getNumSuccessors() > 1
                || std::distance(pred_begin(currBB), pred_end(currBB)) > 1
                )
            {
                currBB->splitBasicBlock(next(endOfSync));
            }

            dbgs()<<"2\n";
            dbgs()<<formatBB(currBB->getSinglePredecessor())<<"\n\n";
            dbgs()<<formatBB(currBB)<<"\n\n";
            dbgs()<<formatBB(currBB->getSingleSuccessor())<<"\n\n";

            return make_tuple(currBB->getSinglePredecessor(), currBB, currBB->getSingleSuccessor());
        }

	void clearBasicBlock(llvm::BasicBlock *bb)
        {
            if (!bb)
                return;

            for (auto it = bb->begin(), end = bb->end(); it != end;)
            {
                if (it->isTerminator())
                {
                    break;
                }
                it = it->eraseFromParent();
            }
        }

        // Check if the given call instruction is a cooperative group symchronization related calls
        // @_ZN18cooperative_groups11synchronizeENS_12thread_blockE
        // @_ZNK18cooperative_groups12thread_group4syncEv(
        bool isSync(Instruction *I)
        {
            bool state = isThreadBlockSync(I) || isThreadGroupSync(I);
            if (state)
            {
                dbgs() << "isSync: sync found \"" << formatInst(I) << "\"\n\n";
            }
            return state;
        }

        bool isThreadBlockSync(Instruction *I)
        {
            CallInst *callInst;
            if (!(callInst = dyn_cast<CallInst>(I)))
            {
                return false;
            }

            auto FuncNameStr = callInst->getCalledFunction()->getName();

            // __syncthreads()
            bool isSyncThreads = FuncNameStr.find("_ZL13__syncthreadsv") != FuncNameStr.npos;

            // cg::
            bool hasCgnamespace = FuncNameStr.find("cooperative_groups") != FuncNameStr.npos;
            // synchronize(thread_block)
            bool hasBlockSynchronize = FuncNameStr.find("synchronize") != FuncNameStr.npos && FuncNameStr.find("thread_block") != FuncNameStr.npos;
            // thread_block::sync()
            bool hasBlockSync = FuncNameStr.find("sync") != FuncNameStr.npos && FuncNameStr.find("thread_block") != FuncNameStr.npos;

            return (hasCgnamespace && (hasBlockSynchronize || hasBlockSync)) || isSyncThreads;
        }

	bool isThreadGroupSync(Instruction *I)
        {
            CallInst *callInst;
            if (!(callInst = dyn_cast<CallInst>(I)))
            {
                return false;
            }

            auto FuncNameStr = callInst->getCalledFunction()->getName();

            // cg::
            bool hasCgnamespace = FuncNameStr.find("cooperative_groups") != FuncNameStr.npos;
            // synchronize(thread_block)
            bool hasBlockSynchronize = FuncNameStr.find("synchronize") != FuncNameStr.npos && FuncNameStr.find("thread_group") != FuncNameStr.npos;
            // thread_block::sync()
            bool hasBlockSync = FuncNameStr.find("sync") != FuncNameStr.npos && FuncNameStr.find("thread_group") != FuncNameStr.npos;

            return hasCgnamespace && (hasBlockSynchronize || hasBlockSync);
        }

        bool isThreadIdxCall(Instruction *I)
        {
            CallInst *callInst;
            if (!(callInst = dyn_cast<CallInst>(I)))
            {
                return false;
            }

            auto FuncNameStr = callInst->getCalledFunction()->getName();
            return FuncNameStr.find("get_local_id") != FuncNameStr.npos;
        }

        // Check if the given call instruction is tiled_partition
        bool isTiledPartition(Instruction *I)
        {

            CallInst *callInst;
            if (!(callInst = dyn_cast<CallInst>(I)))
            {
                return false;
            }

            auto FuncNameStr = callInst->getCalledFunction()->getName();

            if (FuncNameStr.find("cooperative_groups") == FuncNameStr.npos || FuncNameStr.find("tiled_partition") == FuncNameStr.npos || FuncNameStr.find("thread_block") == FuncNameStr.npos)
            {
                return false;
            }

            return true;
        }

	bool isBasicBlockWithinRegion(CodeRegion region, BasicBlock *bb)
        {
            // // Print out region info, bb info
            // dbgs()<<"IsBasicBlockWithinRegion: region: "<<formatBB(region.startBB)<<" - "<<formatBB(region.endBB)<<"\n";
            // dbgs()<<"IsBasicBlockWithinRegion: bb: "<<formatBB(bb)<<"\n\n";
            // // Print out conditions
            // dbgs()<<"IsBasicBlockWithinRegion: conditions: "
            // <<(DT.dominates(region.startBB, bb))
            // <<(PDT.dominates(region.endBB, bb))
            // <<"\n\n";
            return (DT.dominates(region.startBB, bb) && PDT.dominates(region.endBB, bb));
        }

        BasicBlock* findEndOfRegion(BasicBlock *curr, BasicBlock *bound)
        {
            for (auto *pred : llvm::predecessors(bound)) {
                if (DT.dominates(curr, pred)) {
                    return pred;
                }
            }

            return nullptr;
        }

        BasicBlock* findStartOfRegion(BasicBlock *end, BasicBlock *bound)
        {
            for (auto *succ : llvm::successors(bound)) {
                if (DT.dominates(succ, end)) {
                    return succ;
                }
            }

            return nullptr;
        }
    };
}

static bool isSpirKernel(Function &function)
{
    auto state = function.getCallingConv() == CallingConv::SPIR_KERNEL;
    if (state)
    {
        dbgs() << "isSpirKernel: kernel found \"" << function.getName() << "\"\n\n";
        return state;
    }
    return state;
}

PreservedAnalyses HipTaskSyncPass::run(Module &M,
                                       ModuleAnalysisManager &AM)
{
    bool changed = false;
    for (auto &function : M)
    {
        if (isSpirKernel(function))
        {
            changed |= CooperativeGroups::KernelTranslator(function).run();
        }
    }

    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo()
{
    return {LLVM_PLUGIN_API_VERSION, PASS_ID, LLVM_VERSION_STRING,
            [](PassBuilder &PB)
            {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, ModulePassManager &MPM,
                       ArrayRef<PassBuilder::PipelineElement>)
                    {
                        if (Name == PASS_ID)
                        {
                            MPM.addPass(HipTaskSyncPass());
                            return true;
                        }
                        return false;
                    });
            }};
}
                     


