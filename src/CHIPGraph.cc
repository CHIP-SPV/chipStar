/*
 * Copyright (c) 2021-22 chipStar developers
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

#include "CHIPBackend.hh"
#include "CHIPBindingsInternal.hh"

#include <sstream>
// CHIPGraphNode
//*************************************************************************************
void CHIPGraphNode::DFS(std::vector<CHIPGraphNode *> CurrPath,
                        std::vector<std::vector<CHIPGraphNode *>> &Paths) {
  // A node that is already on the path depends on itself through a cycle;
  // the walk would never reach a root.
  if (std::find(CurrPath.begin(), CurrPath.end(), this) != CurrPath.end())
    CHIPERR_LOG_AND_THROW("Graph node " + Msg +
                              " depends on itself through a cycle",
                          hipErrorInvalidValue);
  CurrPath.push_back(this);
  for (auto &Dep : Dependencies_) {
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

   * 1. Use the overriden CHIPGraphNode operator==() to check if two nodes are
   identical
   * 2. Create a map that maps pointers from old graph to new graph
   * 3. Remap the cloned graph.
   *
   */
  std::cout << "\n\n";
  for (CHIPGraphNode *OriginalNode : OriginalGraph.Nodes_) {
    CHIPGraphNode *CloneNode = OriginalNode->clone();
    Nodes_.push_back(CloneNode);
    CloneMap_[OriginalNode] = CloneNode;
    logDebug("Adding to CloneMap: Original {} {} -> Clone {} {}",
             OriginalNode->Msg, (void *)OriginalNode, CloneNode->Msg,
             (void *)CloneNode);
  }

  for (CHIPGraphNode *Node : Nodes_) {
    Node->updateDependencies(CloneMap_);
    Node->updateDependants(CloneMap_);
  }
}

CHIPGraphNodeKernel::CHIPGraphNodeKernel(const CHIPGraphNodeKernel &Other)
    : CHIPGraphNode(Other) {
  // Other.Params_.kernelParams points into Other's own argument buffer.
  // Going through setParams() gives the copy its own argument bytes, exec
  // item and kernel handle, so it does not depend on Other staying alive.
  setParams(Other.Params_);
}

CHIPGraphNode *CHIPGraphNodeKernel::clone() const {
  auto NewNode = new CHIPGraphNodeKernel(*this);
  return NewNode;
}

std::string CHIPGraphNodeKernel::getKernelName() const {
  return ExecItem_->getKernel()->getName();
}

// Defined here rather than in the header: CHIPBackend.hh includes CHIPGraph.hh
// before chipstar::ExecItem is complete, and deleting through an incomplete
// type would skip ExecItem's virtual destructor.
CHIPGraphNodeKernel::~CHIPGraphNodeKernel() { delete ExecItem_; }

void CHIPGraphNodeMemset::execute(chipstar::Queue *Queue) const {
  const unsigned int Val = Params_.value;
  size_t Height = std::max<size_t>(1, Params_.height);
  size_t Width = std::max<size_t>(1, Params_.width);
  size_t Size = Height * Width; //  TODO Graphs Pitch?
  Queue->memFillAsync(Params_.dst, Size, (void *)&Val, Params_.elementSize);
}

void CHIPGraphNodeMemcpy::execute(chipstar::Queue *Queue) const {
  if (Dst_ && Src_) {
    // Use Queue's memCopyAsync to ensure work goes to the correct queue
    Queue->memCopyAsync(Dst_, Src_, Count_, Kind_);
  } else {
    auto Status = hipMemcpy3DAsyncInternal(&Params_, Queue);
    if (Status != hipSuccess)
      CHIPERR_LOG_AND_THROW("Error enountered while executing a graph node",
                            hipErrorTbd);
  }
}
void CHIPGraphNodeKernel::execute(chipstar::Queue *Queue) const {
  // Ensure the kernel module's device variables are allocated before launch.
  // The normal hipLaunchKernel path does this, but graph-node execution
  // bypasses it — which matters when globals are lowered to kernel arguments
  // (their device address must be bound at launch).
  if (auto *K = ExecItem_->getKernel())
    if (const void *HPtr = K->getHostPtr())
      Queue->getDevice()->prepareDeviceVariables(HostPtr(HPtr));
  Queue->launch(ExecItem_);
}

CHIPGraphNodeKernel::CHIPGraphNodeKernel(const hipKernelNodeParams *TheParams)
    : CHIPGraphNode(hipGraphNodeTypeKernel) {
  setParams(*TheParams);
}

CHIPGraphNodeKernel::CHIPGraphNodeKernel(const void *HostFunction, dim3 GridDim,
                                         dim3 BlockDim, void **Args,
                                         size_t SharedMem)
    : CHIPGraphNode(hipGraphNodeTypeKernel) {
  hipKernelNodeParams Params = {};
  Params.func = const_cast<void *>(HostFunction);
  Params.gridDim = GridDim;
  Params.blockDim = BlockDim;
  Params.sharedMemBytes = SharedMem;
  Params.kernelParams = Args;
  Params.extra = nullptr;
  setParams(Params);
}

void CHIPGraphNodeKernel::setParams(const hipKernelNodeParams &Params) {
  auto *Dev = Backend->getActiveDevice();
  chipstar::Kernel *ChipKernel = Dev->findKernel(HostPtr(Params.func));
  if (!ChipKernel)
    CHIPERR_LOG_AND_THROW("Could not find requested kernel",
                          hipErrorInvalidDeviceFunction);

  // Copy into fresh buffers before touching the members: Params may be this
  // node's own getParams() result, whose kernelParams point into ArgData_.
  std::vector<char> ArgData;
  std::vector<void *> ArgList;
  copyKernelArgs(ArgList, ArgData, Params.kernelParams,
                 *ChipKernel->getFuncInfo());
  ArgData_.swap(ArgData);
  ArgList_.swap(ArgList);

  Params_.func = Params.func;
  Params_.gridDim = Params.gridDim;
  Params_.blockDim = Params.blockDim;
  Params_.sharedMemBytes = Params.sharedMemBytes;
  Params_.extra = Params.extra;
  Params_.kernelParams = ArgList_.data();

  delete ExecItem_;
  ExecItem_ = Backend->createExecItem(Params_.gridDim, Params_.blockDim,
                                      Params_.sharedMemBytes, nullptr);
  ExecItem_->setKernel(ChipKernel);
  // Give this graph node a private kernel handle so that another node
  // launching the same kernel does not clobber this node's argument
  // bindings when both are queued before execution (issue #782).
  ExecItem_->useIndependentKernelHandle();
  ExecItem_->setArgs(Params_.kernelParams);
  // setupAllArgs() binds implicit device-global address arguments, so the
  // module's device variables must be allocated first. The normal launch path
  // does this, but a graph node is built before any launch.
  Dev->prepareDeviceVariables(HostPtr(Params_.func));
  ExecItem_->setupAllArgs();
}

static std::string dotEscape(const std::string &Str) {
  std::string Out;
  for (char C : Str) {
    if (C == '"' || C == '\\')
      Out += '\\';
    Out += C;
  }
  return Out;
}

static std::string dim3ToString(dim3 Dim) {
  return "(" + std::to_string(Dim.x) + "," + std::to_string(Dim.y) + "," +
         std::to_string(Dim.z) + ")";
}

static std::string ptrToString(const void *Ptr) {
  std::ostringstream Out;
  Out << Ptr;
  return Out.str();
}

/// Emits the nodes and edges of Graph; child graphs become nested clusters.
static void writeDotBody(CHIPGraph *Graph, std::ostream &Out, unsigned Flags) {
  const bool Verbose = Flags & hipGraphDebugDotFlagsVerbose;
  const bool KernelParams =
      Verbose || (Flags & hipGraphDebugDotFlagsKernelNodeParams);
  const bool MemsetParams =
      Verbose || (Flags & hipGraphDebugDotFlagsMemsetNodeParams);
  const bool HostParams =
      Verbose || (Flags & hipGraphDebugDotFlagsHostNodeParams);
  const bool EventParams =
      Verbose || (Flags & hipGraphDebugDotFlagsEventNodeParams);
  const bool Handles = Verbose || (Flags & hipGraphDebugDotFlagsHandles);

  for (auto *Node : Graph->getNodes()) {
    std::string Label;
    switch (Node->getType()) {
    case hipGraphNodeTypeKernel: {
      auto *Kernel = static_cast<CHIPGraphNodeKernel *>(Node);
      Label = "KERNEL\\n" + dotEscape(Kernel->getKernelName());
      if (KernelParams) {
        auto Params = Kernel->getParams();
        Label += "\\ngrid " + dim3ToString(Params.gridDim) + " block " +
                 dim3ToString(Params.blockDim) + " sharedMem " +
                 std::to_string(Params.sharedMemBytes);
      }
      break;
    }
    case hipGraphNodeTypeMemcpy:
      Label = "MEMCPY";
      break;
    case hipGraphNodeTypeMemset: {
      Label = "MEMSET";
      if (MemsetParams) {
        auto Params = static_cast<CHIPGraphNodeMemset *>(Node)->getParams();
        Label += "\\ndst " + ptrToString(Params.dst) + " value " +
                 std::to_string(Params.value) + " elementSize " +
                 std::to_string(Params.elementSize) + " width " +
                 std::to_string(Params.width) + " height " +
                 std::to_string(Params.height);
      }
      break;
    }
    case hipGraphNodeTypeHost: {
      Label = "HOST";
      if (HostParams) {
        auto Params = static_cast<CHIPGraphNodeHost *>(Node)->getParams();
        Label += "\\nfn " + ptrToString((const void *)Params.fn) +
                 " userData " + ptrToString(Params.userData);
      }
      break;
    }
    case hipGraphNodeTypeGraph:
      Label = "CHILD_GRAPH";
      break;
    case hipGraphNodeTypeEmpty:
      Label = "EMPTY";
      break;
    case hipGraphNodeTypeWaitEvent: {
      Label = "WAIT_EVENT";
      if (EventParams)
        Label += "\\nevent " +
                 ptrToString(
                     static_cast<CHIPGraphNodeWaitEvent *>(Node)->getEvent());
      break;
    }
    case hipGraphNodeTypeEventRecord: {
      Label = "EVENT_RECORD";
      if (EventParams)
        Label +=
            "\\nevent " +
            ptrToString(
                static_cast<CHIPGraphNodeEventRecord *>(Node)->getEvent());
      break;
    }
    case hipGraphNodeTypeMemcpyFromSymbol:
      Label = "MEMCPY_FROM_SYMBOL";
      break;
    case hipGraphNodeTypeMemcpyToSymbol:
      Label = "MEMCPY_TO_SYMBOL";
      break;
    default:
      Label = "NODE_TYPE_" + std::to_string(Node->getType());
      break;
    }
    if (Handles)
      Label += "\\n" + ptrToString(Node);

    Out << "  \"" << ptrToString(Node) << "\" [label=\"" << Label << "\"];\n";

    if (Node->getType() == hipGraphNodeTypeGraph) {
      Out << "  subgraph \"cluster_" << ptrToString(Node)
          << "\" {\n  label=\"CHILD_GRAPH\";\n";
      writeDotBody(static_cast<CHIPGraphNodeGraph *>(Node)->getGraph(), Out,
                   Flags);
      Out << "  }\n";
    }
  }

  for (auto *Node : Graph->getNodes())
    for (auto *Dep : Node->getDependencies())
      Out << "  \"" << ptrToString(Dep) << "\" -> \"" << ptrToString(Node)
          << "\";\n";
}

void CHIPGraph::writeDot(std::ostream &Out, unsigned Flags) {
  Out << "digraph {\n  node [shape=box];\n";
  writeDotBody(this, Out, Flags);
  Out << "}\n";
}

int NodeCounter = 1;
void CHIPGraph::addNode(CHIPGraphNode *Node) {
  logDebug("{} CHIPGraph::addNode({})", (void *)this, (void *)Node);
  Node->Msg = "M" + std::to_string(NodeCounter);
  NodeCounter++;
  Nodes_.push_back(Node);
}

void CHIPGraph::removeNode(CHIPGraphNode *Node) {
  logDebug("{} CHIPGraph::removeNode({})", (void *)this, (void *)Node);

  auto Found = std::find(Nodes_.begin(), Nodes_.end(), Node);
  if (Found == Nodes_.end()) {
    CHIPERR_LOG_AND_THROW(
        "tried to remove the node which was not found in graph", hipErrorTbd);
  } else {
    Nodes_.erase(Found);
  }
}

void CHIPGraphExec::launch(chipstar::Queue *Queue) {
  logDebug("{} CHIPGraphExec::launch({})", (void *)this, (void *)Queue);
  compile();
  auto ExecQueueCopy = ExecQueues_;
  while (ExecQueueCopy.size()) {
    auto Nodes = ExecQueueCopy.front();
    std::string NodesInThisLevel = "";
    for (auto Node : Nodes) {
      NodesInThisLevel += Node->Msg + " ";
    }
    logDebug("Executing nodes: {}", NodesInThisLevel);
    for (auto Node : Nodes) {
      // The schedule is built from the original nodes, but what runs is the
      // node's copy in the compiled graph: it holds the parameters set through
      // hipGraphExec*NodeSetParams and the hipGraphNodeSetEnabled switch, and
      // edits to the original node after instantiation do not reach it. A
      // node the original graph gained after instantiation has no copy and
      // runs as it is. A disabled node behaves like an empty node.
      auto *ExecNode = CompiledGraph_.nodeLookup(Node);
      if (ExecNode && !ExecNode->isEnabled()) {
        logDebug("Skipping disabled {}", Node->Msg);
        continue;
      }
      logDebug("Executing {}", Node->Msg);
      (ExecNode ? ExecNode : Node)->execute(Queue);
      Queue->finish();
    }

    ExecQueueCopy.pop();
  }
}

void unchainUnnecessaryDeps(std::vector<CHIPGraphNode *> Path,
                            std::vector<CHIPGraphNode *> SubPath) {
  assert(Path.size() > SubPath.size());
  std::string PathStr = "";
  for (auto Node : SubPath) {
    PathStr += Node->Msg + " ";
  }
  std::string LongerPathStr = "";
  for (auto Node : Path) {
    LongerPathStr += Node->Msg + " ";
  }
  logDebug("unchainUnnecessaryDeps({}, {})", PathStr, LongerPathStr);

  for (int i = 0; i < SubPath.size(); i++) {
    if (SubPath[i] != Path[i]) {
      // Paths were enumerated before any pruning, so several (Path, SubPath)
      // pairs can single out the same redundant edge; an earlier pair may
      // already have removed it.
      auto Deps = SubPath[i - 1]->getDependencies();
      if (std::find(Deps.begin(), Deps.end(), SubPath[i]) != Deps.end())
        SubPath[i - 1]->removeDependency(SubPath[i]);
      break;
    }
  }
}

std::vector<CHIPGraphNode *> CHIPGraph::getLeafNodes() {
  std::vector<CHIPGraphNode *> LeafNodes;
  for (auto Node : Nodes_) {
    // no other node depends on leaf node.
    if (Node->getDependants().size() == 0)
      LeafNodes.push_back(Node);
  }

  return LeafNodes;
}

void CHIPGraphExec::pruneGraph_() {
  // Prune the executable's own copy of the graph. The caller keeps using the
  // original hipGraph_t, and hipGraphGetEdges on it has to keep reporting the
  // edges the caller added.
  std::vector<CHIPGraphNode *> LeafNodes_ = CompiledGraph_.getLeafNodes();

  for (auto LeafNode : LeafNodes_) {
    // Generate all paths from leaf to root
    std::vector<CHIPGraphNode *> CurrPath;
    std::vector<std::vector<CHIPGraphNode *>> Paths;
    LeafNode->DFS(CurrPath, Paths);

    if (Paths.size() < 2) {
      continue;
    }

    std::sort(Paths.begin(), Paths.end(),
              [](std::vector<CHIPGraphNode *> PathA,
                 std::vector<CHIPGraphNode *> PathB) {
                return PathA.size() > PathB.size();
              });

    for (auto Path : Paths) {
      // convert the current path to a set
      std::set<CHIPGraphNode *> PathSet(Path.begin(), Path.end());

      // Check other paths to see if they are a subset of this (longer) path
      for (auto SubPathIter = Paths.begin(); SubPathIter != Paths.end();
           SubPathIter++) {
        auto SubPath = *SubPathIter;
        // skip if subpath is longer than path
        if (Path.size() <= SubPath.size() || Path == SubPath) {
          continue;
        }

        // convert the other path to a set
        std::set<CHIPGraphNode *> SubPathSet(SubPath.begin(), SubPath.end());
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
        if (std::includes(PathSet.begin(), PathSet.end(), SubPathSet.begin(),
                          SubPathSet.end())) {
          unchainUnnecessaryDeps(Path, SubPath);
        }
      }
    }
  }
}

std::vector<CHIPGraphNode *> CHIPGraph::getRootNodes() {
  std::vector<CHIPGraphNode *> RootNodes;
  for (auto Node : Nodes_) {
    if (Node->getDependencies().size() == 0) {
      RootNodes.push_back(Node);
    }
  }
  return RootNodes;
}

void CHIPGraphExec::compile() {
  // Every launch rebuilds the schedule from scratch; the levels queued by the
  // previous launch would otherwise run again in front of the new ones.
  ExecQueues_ = {};
  pruneGraph_();
  logDebug("{} CHIPGraphExec::compile()", (void *)this);
  std::vector<CHIPGraphNode *> Nodes = OriginalGraph_->getNodes();
  auto RootNodesVec = OriginalGraph_->getRootNodes();
  std::set<CHIPGraphNode *> RootNodes(RootNodesVec.begin(), RootNodesVec.end());
  ExecQueues_.push(RootNodes);
  //  Remove root nodes from the set of nodes
  for (auto Node : RootNodes) {
    Nodes.erase(std::find(Nodes.begin(), Nodes.end(), Node));
  }

  /**
   * This piece of code will generate sets of nodes that can be executed in
   * parallel. These sets are accumulated into the execution queue. The
   * execution queue starts with the root nodes. To fill the execution queue, we
   * find all the nodes that depend only the nodes in the back of the exec
   * queue.
   */
  std::set<CHIPGraphNode *> NextSet;
  std::set<CHIPGraphNode *> PrevLevelNodes = RootNodes;
  auto NodeIter = Nodes.begin();
  while (Nodes.size()) { // while more unnasigned nodes available
    auto CurrentNodeDeps = (*NodeIter)->getDependencies();
    std::string CurrentNodeDepsStr = "";
    for (auto Node : CurrentNodeDeps) {
      CurrentNodeDepsStr += Node->Msg + " ";
    }
    logDebug("CurrentNode {} Deps: {}", (*NodeIter)->Msg, CurrentNodeDepsStr);
    std::string PrevLevelNodesStr = "";
    for (auto Node : PrevLevelNodes) {
      PrevLevelNodesStr += Node->Msg + " ";
    }
    logDebug("PrevLevelNodes: {}", PrevLevelNodesStr);

    // std::includes requires sorted ranges. Since PrevLevelNodes is a sorted
    // set, we only need to sort the CurrentNodeDeps
    std::sort(CurrentNodeDeps.begin(), CurrentNodeDeps.end());
    if (std::includes(PrevLevelNodes.begin(), PrevLevelNodes.end(),
                      CurrentNodeDeps.begin(), CurrentNodeDeps.end())) {
      NextSet.insert(*NodeIter);
      Nodes.erase(NodeIter);
      NodeIter = Nodes.begin();
    } else {
      NodeIter++;
    }

    if (NodeIter == Nodes.end()) {
      // A pass that places no node would repeat forever: every remaining node
      // waits on a node outside this graph or on a node that is itself still
      // waiting (a cycle, or a graph without a root).
      if (NextSet.empty())
        CHIPERR_LOG_AND_THROW("Graph node " + Nodes.front()->Msg +
                                  " depends on a node that can never run",
                              hipErrorInvalidValue);
      PrevLevelNodes.insert(NextSet.begin(), NextSet.end());
      ExecQueues_.push(NextSet);
      NextSet.clear();
      NodeIter = Nodes.begin();
    }
  }
}

void CHIPGraphNodeHost::execute(chipstar::Queue *Queue) const {
  Queue->finish();
  Params_.fn(Params_.userData);
}

void CHIPGraphNodeGraph::execute(chipstar::Queue *Queue) const {
  // The schedule runs this node after all of its dependencies and before all
  // of its dependants, so running the child graph to completion here keeps
  // the ordering the parent graph asked for.
  CHIPGraphExec SubGraphExec(SubGraph_);
  SubGraphExec.launch(Queue);
}

void CHIPGraphNodeEventRecord::execute(chipstar::Queue *Queue) const {
  NULLCHECK(Event_);
  auto Status = hipEventRecordInternal(Event_, Queue);
  if (Status != hipSuccess)
    CHIPERR_LOG_AND_THROW("Error enountered while executing a graph node",
                          hipErrorTbd);
}

void CHIPGraphNodeMemcpyFromSymbol::execute(chipstar::Queue *Queue) const {
  NULLCHECK(Dst_, Symbol_);
  auto Status = hipMemcpyFromSymbolAsyncInternal(Dst_, Symbol_, SizeBytes_,
                                                 Offset_, Kind_, Queue);
  if (Status != hipSuccess)
    CHIPERR_LOG_AND_THROW("Error enountered while executing a graph node",
                          hipErrorTbd);
}

void CHIPGraphNodeMemcpyToSymbol::execute(chipstar::Queue *Queue) const {
  NULLCHECK(Symbol_, Src_);
  auto Status = hipMemcpyToSymbolAsyncInternal(Symbol_, Src_, SizeBytes_,
                                               Offset_, Kind_, Queue);
  if (Status != hipSuccess)
    CHIPERR_LOG_AND_THROW("Error enountered while executing a graph node",
                          hipErrorTbd);
}

void CHIPGraphNodeWaitEvent::execute(chipstar::Queue *Queue) const {
  // current HIP API requires Flags
  unsigned int Flags = 0;
  auto Status = hipStreamWaitEventInternal(Queue, Event_, Flags);
  if (Status != hipSuccess)
    CHIPERR_LOG_AND_THROW("Error enountered while executing a graph node",
                          hipErrorTbd);
}
