/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
/**
 * @file CHIPGraph.cc
 * @author Paulius Velesko (pvelesko@pglc.io)
 * @brief CHIPGraph Implementation File
 * @version 0.1
 * @date 2022-11-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "CHIPGraph.hh"
// CHIPGraph
//*************************************************************************************
void CHIPGraphNode::DFS(std::vector<CHIPGraphNode*> CurrPath, std::vector<std::vector<CHIPGraphNode*>> &Paths) {
          CurrPath.push_back(this);
      for (auto & Dep : Dependencies_) {
          Dep->DFS(CurrPath, Paths);
      }
      
      if (Dependencies_.size() == 0) {
          Paths.push_back(CurrPath);
          // std::string PathStr = "";
          // for(auto & Node : CurrPath) {
          //     PathStr += Node->Msg + ", ";
          // }
          // logDebug("PATH: {}", PathStr);
      }
      
      CurrPath.pop_back();
      return;
  }

  CHIPGraph::CHIPGraph(const CHIPGraph &OriginalGraph) {
    /**
     * Create another Graph using the copy constructor. 
     * This other graph will contain vectors/sets for dependencies/edges. 
     * The edges, however, are pointers which point nodes in the original graph. 
     * These edges must be remapped to this node. 

     * 1. Use the overriden CHIPGraphNode operator==() to check if two nodes are identical
     * 2. Create a map that maps pointers from old graph to new graph
     * 3. Remap the cloned graph. 
     * 
     */
    std::cout << "\n\n";
    for (CHIPGraphNode* OriginalNode : OriginalGraph.Nodes_) {
      CHIPGraphNode* CloneNode = OriginalNode->clone();
      Nodes_.push_back(CloneNode);
      CloneMap_[OriginalNode] = CloneNode;
      logDebug("Adding to CloneMap: Original {} {} -> Clone {} {}", OriginalNode->Msg, (void*)OriginalNode, CloneNode->Msg, (void*)CloneNode);
    }

    for(CHIPGraphNode* Node : Nodes_) {
        Node->updateDependencies(CloneMap_);
        Node->updateDependants(CloneMap_);
      }


  }

  CHIPGraphNodeKernel::CHIPGraphNodeKernel(const CHIPGraphNodeKernel &Other) : CHIPGraphNode(Other) {
    Params_ = Other.Params_;
    ExecItem_ = Other.ExecItem_->clone();
  }

    CHIPGraphNode* CHIPGraphNodeKernel::clone() const {
    auto NewNode = new CHIPGraphNodeKernel(*this);
    return NewNode;
  }


void CHIPGraphNodeMemset::execute(CHIPQueue *Queue) const {
  const unsigned int Val = Params_.value;
  size_t Height = std::max<size_t>(1, Params_.height);
  size_t Width = std::max<size_t>(1, Params_.width); 
  size_t Size = Height * Width; //  TODO Graphs Pitch?
  Queue->memFillAsync(Params_.dst, Size, (void*)&Val, Params_.elementSize);
}

void CHIPGraphNodeMemcpy::execute(CHIPQueue* Queue) const {
  hipMemcpy3DAsync(&Params_, Queue);
}
void CHIPGraphNodeKernel::execute(CHIPQueue* Queue) const {
    Queue->launch(ExecItem_);
  }

CHIPGraphNodeKernel::CHIPGraphNodeKernel(const hipKernelNodeParams * TheParams) : CHIPGraphNode() {
  Type_ = hipGraphNodeTypeKernel;
  Params_.blockDim = TheParams->blockDim;
  Params_.extra = TheParams->extra;
  Params_.func = TheParams->func;
  Params_.gridDim = TheParams->gridDim;
  Params_.kernelParams = TheParams->kernelParams;
  Params_.sharedMemBytes = TheParams->sharedMemBytes;
  auto Dev = Backend->getActiveDevice();
  CHIPKernel *ChipKernel = Dev->findKernelByHostPtr(Params_.func);
  ExecItem_ = Backend->createCHIPExecItem(Params_.gridDim, Params_.blockDim, Params_.sharedMemBytes, nullptr);
  ExecItem_->setArgPointer(Params_.kernelParams);
  ExecItem_->setKernel(ChipKernel);

  ExecItem_->copyArgs(TheParams->kernelParams);
  ExecItem_->setupAllArgs();
  }

CHIPGraphNodeKernel::CHIPGraphNodeKernel(const void *HostFunction, dim3 GridDim,
                           dim3 BlockDim, void **Args, size_t SharedMem)  {
  Type_ = hipGraphNodeTypeKernel;
  Params_.blockDim = BlockDim;
  Params_.extra = nullptr;
  Params_.func = const_cast<void*>(HostFunction); // TODO Graphs why can't I assign const void* to void*?
  Params_.gridDim = GridDim;
  Params_.kernelParams = Args;
  Params_.sharedMemBytes = SharedMem;

  // TODO Graphs use Graph object as factory and get the device this way
  auto Dev = Backend->getActiveDevice();
  CHIPKernel *ChipKernel = Dev->findKernelByHostPtr(HostFunction);
  ExecItem_ = Backend->createCHIPExecItem(GridDim, BlockDim, SharedMem, nullptr);
  ExecItem_->setArgPointer(Args);
  ExecItem_->setKernel(ChipKernel);

  ExecItem_->copyArgs(Args);
  ExecItem_->setupAllArgs();
  }

int NodeCounter = 1;
void CHIPGraph::addNode(CHIPGraphNode* Node) {
  logDebug("{} CHIPGraph::addNode({})", (void*)this, (void*)Node);
  Node->Msg = "M" + std::to_string(NodeCounter);
  NodeCounter++;
  Nodes_.push_back(Node);
}

void CHIPGraph::removeNode(CHIPGraphNode* Node) {
  logDebug("{} CHIPGraph::removeNode({})", (void*)this, (void*)Node);

  auto Found = std::find(Nodes_.begin(), Nodes_.end(), Node);
  if (Found == Nodes_.end()) {
    CHIPERR_LOG_AND_THROW("tried to remove the node which was not found in graph", hipErrorTbd);
  } else {
    Nodes_.erase(Found);
  }
}

void CHIPGraphExec::launch(CHIPQueue *Queue) {
  logDebug("{} CHIPGraphExec::launch({})", (void*)this, (void*)Queue);
  compile();
  auto ExecQueueCopy = ExecQueues_;
  while(ExecQueueCopy.size()) {
    auto Nodes = ExecQueueCopy.front();
    std::string NodesInThisLevel = "";
    for (auto Node : Nodes) {
      NodesInThisLevel += Node->Msg + " ";
    }
    logDebug("Executing nodes: {}", NodesInThisLevel);
    for(auto Node : Nodes) {
      logDebug("Executing {}", Node->Msg);
      Node->execute(Queue);
      Queue->finish();
    }
    
    ExecQueueCopy.pop();
  }

}

void unchainUnnecessaryDeps(std::vector<CHIPGraphNode*> Path, std::vector<CHIPGraphNode*> SubPath) {
  assert(Path.size() > SubPath.size());
  std::string PathStr = ""; 
  for(auto Node : SubPath) {
    PathStr += Node->Msg + " ";
  }
  std::string LongerPathStr = "";
  for(auto Node : Path) {
    LongerPathStr += Node->Msg + " ";
  }
  logDebug("unchainUnnecessaryDeps({}, {})", PathStr, LongerPathStr);

  for (int i = 0; i < SubPath.size(); i++) {
    if(SubPath[i] != Path[i]) {
      SubPath[i-1]->removeDependency(SubPath[i]);
      break;
    }
  }
}

std::vector<CHIPGraphNode*> CHIPGraph::getLeafNodes() {
  std::vector<CHIPGraphNode*> LeafNodes;
  for(auto Node : Nodes_) {
    // no other node depends on leaf node.
    bool LeafNode = true;
    for(auto OtherNode : Nodes_) {
      if (OtherNode->getDependenciesSet().count(Node)) {
        LeafNode = false;
        break;
      }
    }
    if(LeafNode) {
      LeafNodes.push_back(Node);
    }
  }

  return LeafNodes;
}

void CHIPGraphExec::pruneGraph() {
  Pruned_ = true;
  std::vector<CHIPGraphNode*> LeafNodes_ = Graph_->getLeafNodes();

  for(auto LeafNode : LeafNodes_) {
  // Generate all paths from leaf to root
  std::vector<CHIPGraphNode*> CurrPath;
  std::vector<std::vector<CHIPGraphNode*>> Paths;
  LeafNode->DFS(CurrPath, Paths);

  if(Paths.size() < 2) {
    continue;
  }

  std::sort(Paths.begin(), Paths.end(), [](std::vector<CHIPGraphNode*> PathA, std::vector<CHIPGraphNode*> PathB) {
    return PathA.size() > PathB.size();
  } );

  for(auto Path : Paths){
    // convert the current path to a set
    std::set<CHIPGraphNode*> PathSet(Path.begin(), Path.end());

    // Check other paths to see if they are a subset of this (longer) path
    for(auto SubPathIter = Paths.begin(); SubPathIter!= Paths.end(); SubPathIter++) {
      auto SubPath = *SubPathIter;
      // skip if subpath is longer than path
      if (Path.size() <= SubPath.size() || Path == SubPath) {
        continue;
      }

      // convert the other path to a set
      std::set<CHIPGraphNode*> SubPathSet(SubPath.begin(), SubPath.end());
      // std::string PathStr = "";
      // for(auto Node : Path) {
      //   PathStr += Node->Msg + " ";
      // }
      // std::string SubPathStr = "";
      // for(auto Node : SubPath) {
      //   SubPathStr += Node->Msg + " ";
      // }
      // logDebug("Path: {}", PathStr);
      // logDebug("OtherPath: {}", SubPathStr);
      if(std::includes(PathSet.begin(), PathSet.end(), SubPathSet.begin(), SubPathSet.end())) {
        unchainUnnecessaryDeps(Path, SubPath);
      }
    }
  }
  }


}

std::vector<CHIPGraphNode*> CHIPGraph::getRootNodes() {
  std::vector<CHIPGraphNode*> RootNodes;
  for(auto Node : Nodes_) {
    if (Node->getDependenciesSet().size() == 0) {
      RootNodes.push_back(Node);
    }
  }
  return RootNodes;
}

void CHIPGraphExec::compile() {
  pruneGraph();
  logDebug("{} CHIPGraphExec::compile()", (void*)this);
  std::vector<CHIPGraphNode*> Nodes = Graph_->getNodes();
  auto RootNodesVec = Graph_->getRootNodes();
  std::set<CHIPGraphNode*> RootNodes(RootNodesVec.begin(), RootNodesVec.end());
  ExecQueues_.push(RootNodes);
  //  Remove root nodes from the set of nodes
  for(auto Node : RootNodes) {
    Nodes.erase(std::find(Nodes.begin(), Nodes.end(), Node));
  }

  /**
   * This piece of code will generate sets of nodes that can be executed in parallel.
   * These sets are accumulated into the execution queue. The execution queue starts with the root nodes.
   * To fill the execution queue, we find all the nodes that depend only the nodes in the back of the exec queue. 
   */
  std::set<CHIPGraphNode*> NextSet;
  std::set<CHIPGraphNode*> PrevLevelNodes = RootNodes;
  auto NodeIter = Nodes.begin();
  while(Nodes.size()) { // while more unnasigned nodes available
    const std::set<CHIPGraphNode*> CurrentNodeDeps = (*NodeIter)->getDependenciesSet();
    std::string CurrentNodeDepsStr = "";
    for(auto Node : CurrentNodeDeps) {
      CurrentNodeDepsStr += Node->Msg + " ";
    }
    // logDebug("CurrentNode {} Deps: {}", (*NodeIter)->Msg, CurrentNodeDepsStr);
    // std::string PrevLevelNodesStr = "";
    // for(auto Node : PrevLevelNodes) {
    //   PrevLevelNodesStr += Node->Msg + " ";
    // }
    // logDebug("PrevLevelNodes: {}", PrevLevelNodesStr);

    if(std::includes(PrevLevelNodes.begin(), PrevLevelNodes.end(), CurrentNodeDeps.begin(), CurrentNodeDeps.end())) {
      NextSet.insert(*NodeIter);
      Nodes.erase(NodeIter);
      NodeIter = Nodes.begin();
    } else {
      NodeIter++;
    }

    if(NodeIter == Nodes.end()) {
      PrevLevelNodes.insert(NextSet.begin(), NextSet.end());
      ExecQueues_.push(NextSet);
      NextSet.clear();
      NodeIter = Nodes.begin();
    }
  }

}