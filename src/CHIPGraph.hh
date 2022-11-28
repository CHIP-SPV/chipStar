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
 * @file CHIPGraph.hh
 * @author Paulius Velesko (pvelesko@pglc.io)
 * @brief CHIPGraph Header File
 * @version 0.1
 * @date 2022-11-28
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef CHIP_GRAPH_H
#define CHIP_GRAPH_H

#include "common.hh"
#include "CHIPBackend.hh"
#include "hip/hip_runtime_api.h"
#include "CHIPException.hh"
#include "logging.hh"
#include "macros.hh"

class CHIPGraphNode {
protected:
  hipGraphNodeType Type_;
  // nodes which depend on this node
  std::set<CHIPGraphNode *> Dependendants_;
  // nodes on which this node depends
  std::set<CHIPGraphNode *> Dependencies_;
  /**
   * @brief Destroy the CHIPGraphNode object
   * Hidden virtual destructor. Should only be called through derived classes.
   */
  virtual ~CHIPGraphNode() {
    Dependendants_.clear();
    Dependencies_.clear();
  }

public:
  std::string Msg; // TODO Graphs cleanup
  CHIPGraphNode() {}
  CHIPGraphNode(const CHIPGraphNode &Other)
      : Type_(Other.Type_), Dependendants_(Other.Dependendants_),
        Dependencies_(Other.Dependencies_), Msg(Other.Msg) {}
  virtual bool operator==(const CHIPGraphNode &Other) const = 0;

  hipGraphNodeType getType() { return Type_; }
  virtual CHIPGraphNode *clone() const = 0;

  /**
   * @brief Depth-first search of the graph.
   * For each node, the function is called recursively on all of its
   * dependencies. Returns all possible paths from this node to root node.
   *
   * @param CurrPath space for the current path
   * @param Paths space for the paths
   */
  void DFS(std::vector<CHIPGraphNode *> CurrPath,
           std::vector<std::vector<CHIPGraphNode *>> &Paths);
  virtual void execute(CHIPQueue *Queue) const = 0;

  void addDependant(CHIPGraphNode *TheNode) {
    logDebug("{} addDependant() <{} depends on {}>", (void *)this, TheNode->Msg,
             Msg);
    Dependendants_.insert(TheNode);
  }

  void addDependency(CHIPGraphNode *TheNode) {
    logDebug("{} addDependency() <{} depends on {}>", (void *)this, Msg,
             TheNode->Msg);
    Dependencies_.insert(TheNode);
    TheNode->addDependant(this);
  }

  void removeDependency(CHIPGraphNode *TheNode) {
    logDebug("{} removeDependency() <{} depends on {}>", (void *)this, Msg,
             TheNode->Msg);
    Dependencies_.erase(TheNode);
  }

  void addDependencies(CHIPGraphNode **Dependencies, int Count) {
    for (int i = 0; i < Count; i++) {
      addDependency(Dependencies[i]);
    }
  }

  void removeDependencies(CHIPGraphNode **Dependencies, int Count) {
    for (int i = 0; i < Count; i++) {
      removeDependency(Dependencies[i]);
    }
  }

  void updateDependencies(std::map<CHIPGraphNode *, CHIPGraphNode *> CloneMap) {
    std::set<CHIPGraphNode *> NewDeps;
    for (auto Dep : Dependencies_) {
      auto ClonedDep = CloneMap[Dep];
      logDebug("{} {} Replacing dependency {} with {}", (void *)this, this->Msg,
               (void *)Dep, (void *)ClonedDep);
      NewDeps.insert(ClonedDep);
    }
    Dependencies_.clear();
    Dependencies_ = NewDeps;
    return;
  }

  void updateDependants(std::map<CHIPGraphNode *, CHIPGraphNode *> CloneMap) {
    std::set<CHIPGraphNode *> NewDeps;
    for (auto Dep : Dependendants_) {
      auto ClonedDep = CloneMap[Dep];
      logDebug("{} {} Replacing dependant {} with {}", (void *)this, this->Msg,
               (void *)Dep, (void *)ClonedDep);
      NewDeps.insert(ClonedDep);
    }
    Dependendants_.clear();
    Dependendants_ = NewDeps;
    return;
  }

  /**
   * @brief get the nodes on which this node depends on.
   *
   * @return const std::set<CHIPGraphNode*>&
   */
  const std::set<CHIPGraphNode *> &
  getDependenciesSet() { // TODO Graphs prob just use vectors
    return Dependencies_;
  }

  std::vector<CHIPGraphNode *> getDependenciesVec() const {
    std::vector<CHIPGraphNode *> Deps;
    auto DepsIter = Dependencies_.begin();
    while (DepsIter != Dependencies_.end()) {
      CHIPGraphNode *DepNode = *DepsIter;
      Deps.push_back(DepNode);
      DepsIter++;
    }
    return Deps;
  }

  /**
   * @brief get the nodes on which this node depends on.
   *
   * @return const std::set<CHIPGraphNode*>&
   */
  const std::set<CHIPGraphNode *> &getDependantsSet() { return Dependendants_; }

  std::vector<CHIPGraphNode *> getDependantsVec() const {
    std::vector<CHIPGraphNode *> Deps;
    auto DepsIter = Dependendants_.begin();
    while (DepsIter != Dependendants_.end()) {
      CHIPGraphNode *DepNode = *DepsIter;
      Deps.push_back(DepNode);
      DepsIter++;
    }
    return Deps;
  }
};

class CHIPGraphNodeKernel : public CHIPGraphNode {
private:
  hipKernelNodeParams Params_;
  CHIPExecItem *ExecItem_;

public:
  CHIPGraphNodeKernel(const CHIPGraphNodeKernel &Other);

  CHIPGraphNodeKernel(const hipKernelNodeParams *TheParams);

  CHIPGraphNodeKernel(const void *HostFunction, dim3 GridDim, dim3 BlockDim,
                      void **Args, size_t SharedMem);
  virtual void execute(CHIPQueue *Queue) const override;
  hipKernelNodeParams getParams() const { return Params_; }

  void setParams(const hipKernelNodeParams Params) { Params_ = Params; }
  /**
   * @brief Createa a copy of this node
   * Must copy over all the arguments
   * Must copy over all the dependencies.
   * Copying over the dependencies is important because CHIPGraph::clone() uses
   * them to remap onto new nodes
   *
   * @return CHIPGraphNode*
   */
  virtual CHIPGraphNode *clone() const override;

  /**
   * @brief Comparison operator.
   * Must check that data/arguments are the same between two nodes.
   * Must NOT check if the dependencies are the same.
   *
   * @param Other
   * @return true
   * @return false
   */
  virtual bool operator==(const CHIPGraphNode &Other) const override {
    UNIMPLEMENTED(false); // TODO Graphs
  }
};

class CHIPGraphNodeMemcpy : public CHIPGraphNode {
private:
  hipMemcpy3DParms Params_;

public:
  CHIPGraphNodeMemcpy(const CHIPGraphNodeMemcpy &Other)
      : CHIPGraphNode(Other), Params_(Other.Params_) {}

  CHIPGraphNodeMemcpy(hipMemcpy3DParms Params) : Params_(Params) {
    Type_ = hipGraphNodeTypeMemcpy;
  }
  CHIPGraphNodeMemcpy(const hipMemcpy3DParms *Params) {
    Type_ = hipGraphNodeTypeMemcpy;
    setParams(Params);
  }
  hipMemcpy3DParms getParams() { return Params_; }
  void setParams(const hipMemcpy3DParms *Params) {
    Params_.srcArray = Params->srcArray;
    // if(Params->srcArray)
    // memcpy(Params_.srcArray, Params->srcArray, sizeof(hipArray_t));
    memcpy(&Params_.srcPos, &(Params->srcPos), sizeof(hipPos));
    memcpy(&Params_.srcPtr, &(Params->srcPtr), sizeof(hipPitchedPtr));
    Params_.dstArray = Params->dstArray;
    // if(Params->dstArray)
    //   memcpy(Params_.dstArray, Params->dstArray, sizeof(hipArray_t));
    memcpy(&Params_.dstPos, &(Params->dstPos), sizeof(hipPos));
    memcpy(&Params_.dstPtr, &(Params->dstPtr), sizeof(hipPitchedPtr));
    memcpy(&Params_.extent, &(Params->extent), sizeof(hipExtent));
    memcpy(&Params_.kind, &(Params->kind), sizeof(hipMemcpyKind));
  }

  virtual void execute(CHIPQueue *Queue) const override;

  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeMemcpy(*this);
    return NewNode;
  }
  virtual bool operator==(const CHIPGraphNode &Other) const override {
    UNIMPLEMENTED(false); // TODO Graphs
  }
};

class CHIPGraphNodeMemset : public CHIPGraphNode {
private:
  hipMemsetParams Params_;

public:
  CHIPGraphNodeMemset(const CHIPGraphNodeMemset &Other)
      : CHIPGraphNode(Other), Params_(Other.Params_) {}
  CHIPGraphNodeMemset(const hipMemsetParams Params) : Params_(Params) {
    Type_ = hipGraphNodeTypeMemset;
  }
  CHIPGraphNodeMemset(const hipMemsetParams *Params) : Params_(*Params) {
    Type_ = hipGraphNodeTypeMemset;
  }
  hipMemsetParams getParams() { return Params_; }
  void setParams(const hipMemsetParams *Params) { Params_ = *Params; }

  virtual void execute(CHIPQueue *Queue) const override;
  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeMemset(*this);
    return NewNode;
  }
  virtual bool operator==(const CHIPGraphNode &Other) const override {
    UNIMPLEMENTED(false); // TODO Graphs
  }
};

class CHIPGraphNodeHost : public CHIPGraphNode {
private:
  hipHostNodeParams Params_;

public:
  CHIPGraphNodeHost(const CHIPGraphNodeHost &Other)
      : CHIPGraphNode(Other), Params_(Other.Params_) {}
  CHIPGraphNodeHost(const hipHostNodeParams *Params) { Params_ = *Params; }
  virtual void execute(CHIPQueue *Queue) const override {
    // TODO Graphs
    UNIMPLEMENTED();
  }
  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeHost(*this);
    return NewNode;
  }
  virtual bool operator==(const CHIPGraphNode &Other) const override {
    UNIMPLEMENTED(false); // TODO Graphs
  }

  void setParams(const hipHostNodeParams *Params) { Params_ = *Params; }

  hipHostNodeParams getParams() { return Params_; }
};

class CHIPGraphNodeGraph : public CHIPGraphNode {
private:
  CHIPGraph *SubGraph_;

public:
  CHIPGraphNodeGraph(CHIPGraph *Graph) : SubGraph_(Graph) {}
  CHIPGraphNodeGraph(const CHIPGraphNodeGraph &Other)
      : CHIPGraphNode(Other), SubGraph_(Other.SubGraph_) {}

  virtual void execute(CHIPQueue *Queue) const override {
    // TODO Graphs - graph compile step should replace this node with subgraph
    // nodes
    UNIMPLEMENTED();
  }
  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeGraph(*this);
    return NewNode;
  }
  virtual bool operator==(const CHIPGraphNode &Other) const override {
    UNIMPLEMENTED(false); // TODO Graphs
  }

  void setGraph(CHIPGraph *Graph) { SubGraph_ = Graph; }

  CHIPGraph *getGraph() { return SubGraph_; }
};

class CHIPGraphNodeEmpty : public CHIPGraphNode {
public:
  CHIPGraphNodeEmpty(const CHIPGraphNodeEmpty &Other) : CHIPGraphNode(Other) {}
  CHIPGraphNodeEmpty() { Type_ = hipGraphNodeTypeEmpty; };

  virtual void execute(CHIPQueue *Queue) const override {
    logDebug("Executing empty node");
  }

  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeEmpty(*this);
    return NewNode;
  }
  virtual bool operator==(const CHIPGraphNode &Other) const override {
    UNIMPLEMENTED(false); // TODO Graphs
  }
};

class CHIPGraphNodeWaitEvent : public CHIPGraphNode {
private:
  CHIPEvent *Event_;

public:
  CHIPGraphNodeWaitEvent(CHIPEvent *Event) : Event_(Event) {}
  CHIPGraphNodeWaitEvent(const CHIPGraphNodeWaitEvent &Other)
      : CHIPGraphNode(Other), Event_(Other.Event_) {}
  virtual void execute(CHIPQueue *Queue) const override {
    // TODO Graphs current HIP API requires this to be 0
    hipStreamWaitEvent(Queue, Event_, 0);
  }
  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeWaitEvent(*this);
    return NewNode;
  }
  virtual bool operator==(const CHIPGraphNode &Other) const override {
    UNIMPLEMENTED(false); // TODO Graphs
  }

  CHIPEvent *getEvent() { return Event_; }
  void setEvent(CHIPEvent *Event) { Event_ = Event; }
};

class CHIPGraphNodeEventRecord : public CHIPGraphNode {
private:
  CHIPEvent *Event_;

public:
  CHIPGraphNodeEventRecord(CHIPEvent *Event) : Event_(Event){};

  CHIPGraphNodeEventRecord(const CHIPGraphNodeEventRecord &Other)
      : CHIPGraphNode(Other), Event_(Other.Event_) {}
  virtual void execute(CHIPQueue *Queue) const override {
    hipEventRecord(Event_, Queue);
  }
  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeEventRecord(*this);
    return NewNode;
  }
  virtual bool operator==(const CHIPGraphNode &Other) const override {
    UNIMPLEMENTED(false); // TODO Graphs
  }

  void setEvent(CHIPEvent *NewEvent) { Event_ = NewEvent; }

  CHIPEvent *getEvent() { return Event_; }
};

class CHIPGraphNodeMemcpy1D : public CHIPGraphNode {
private:
  void *Dst_;
  const void *Src_;
  size_t Count_;
  hipMemcpyKind Kind_;

public:
  CHIPGraphNodeMemcpy1D(const CHIPGraphNodeMemcpy1D &Other)
      : CHIPGraphNode(Other), Dst_(Other.Dst_), Src_(Other.Src_),
        Count_(Other.Count_), Kind_(Other.Kind_) {}

  CHIPGraphNodeMemcpy1D(void *Dst, const void *Src, size_t Count,
                        hipMemcpyKind Kind)
      : Dst_(Dst), Src_(Src), Count_(Count), Kind_(Kind) {
    Type_ = hipGraphNodeTypeMemcpy1D;
  }

  void setParams(void *Dst, const void *Src, size_t Count, hipMemcpyKind Kind) {
    Dst_ = Dst;
    Src_ = Src;
    Count_ = Count;
    Kind_ = Kind;
  }

  virtual void execute(CHIPQueue *Queue) const override {
    hipMemcpy(Dst_, Src_, Count_, Kind_);
  }
  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeMemcpy1D(*this);
    return NewNode;
  }
  virtual bool operator==(const CHIPGraphNode &Other) const override {
    UNIMPLEMENTED(false);
  }
};

class CHIPGraphNodeMemcpyFromSymbol : public CHIPGraphNode {
private:
  void *Dst_;
  void *Symbol_;
  size_t SizeBytes_;
  size_t Offset_;
  hipMemcpyKind Kind_;

public:
  CHIPGraphNodeMemcpyFromSymbol(void *Dst, const void *Symbol, size_t SizeBytes,
                                size_t Offset, hipMemcpyKind Kind)
      : Dst_(Dst), Symbol_(const_cast<void *>(Symbol)), SizeBytes_(SizeBytes),
        Offset_(Offset), Kind_(Kind) {}

  CHIPGraphNodeMemcpyFromSymbol(const CHIPGraphNodeMemcpyFromSymbol &Other)
      : CHIPGraphNode(Other), Dst_(Other.Dst_), Symbol_(Other.Symbol_),
        SizeBytes_(Other.SizeBytes_), Offset_(Other.Offset_),
        Kind_(Other.Kind_) {}

  virtual void execute(CHIPQueue *Queue) const override {
    hipMemcpyFromSymbol(Dst_, Symbol_, SizeBytes_, Offset_, Kind_);
  }

  void setParams(void *Dst, const void *Symbol, size_t SizeBytes, size_t Offset,
                 hipMemcpyKind Kind) {
    Dst_ = Dst;
    Symbol_ = const_cast<void *>(Symbol);
    SizeBytes_ = SizeBytes;
    Offset_ = Offset;
    Kind = Kind_;
  }

  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeMemcpyFromSymbol(*this);
    return NewNode;
  }

  virtual bool operator==(const CHIPGraphNode &Other) const override {
    UNIMPLEMENTED(false); // TODO Graphs
  }
};

class CHIPGraphNodeMemcpyToSymbol : public CHIPGraphNode {
private:
  void *Src_;
  void *Symbol_;
  size_t SizeBytes_;
  size_t Offset_;
  hipMemcpyKind Kind_;

public:
  CHIPGraphNodeMemcpyToSymbol(void *Src, const void *Symbol, size_t SizeBytes,
                              size_t Offset, hipMemcpyKind Kind)
      : Src_(Src), Symbol_(const_cast<void *>(Symbol)), SizeBytes_(SizeBytes),
        Offset_(Offset), Kind_(Kind) {}

  CHIPGraphNodeMemcpyToSymbol(const CHIPGraphNodeMemcpyToSymbol &Other)
      : CHIPGraphNode(Other), Src_(Other.Src_), Symbol_(Other.Symbol_),
        SizeBytes_(Other.SizeBytes_), Offset_(Other.Offset_),
        Kind_(Other.Kind_) {}

  virtual void execute(CHIPQueue *Queue) const override {
    hipMemcpyToSymbol(Symbol_, Src_, SizeBytes_, Offset_, Kind_);
  }
  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeMemcpyToSymbol(*this);
    return NewNode;
  }

  virtual bool operator==(const CHIPGraphNode &Other) const override {
    UNIMPLEMENTED(false); // TODO Graphs
  }

  void setParams(void *Src, const void *Symbol, size_t SizeBytes, size_t Offset,
                 hipMemcpyKind Kind) {
    Src_ = Src;
    Symbol_ = const_cast<void *>(Symbol);
    SizeBytes_ = SizeBytes;
    Offset_ = Offset;
    Kind = Kind_;
  }
};

class CHIPGraph {
protected:
  CHIPDevice *ChipDev_;
  std::vector<CHIPGraphNode *> Nodes_;
  // Map the pointers Original -> Clone
  std::map<CHIPGraphNode *, CHIPGraphNode *> CloneMap_;

public:
  CHIPGraph(const CHIPGraph &OriginalGraph);
  CHIPGraph(CHIPDevice *ChipDev) : ChipDev_(ChipDev) {}
  void addNode(CHIPGraphNode *TheNode);
  void removeNode(CHIPGraphNode *TheNode);
  /**
   * @brief Lookup a cloned(instantiated) node using a pointer to the original
   * node
   *
   * @param OriginalNode pointer to the node which was present in CHIPGraph at
   * the time of instantiation of CHIPGraph to CHIPGraphExec
   * @return CHIPGraphNode* pointer to the resulting node in CHIPGraphExec which
   * corresponds to the original node
   */
  CHIPGraphNode *nodeLookup(CHIPGraphNode *OriginalNode) {
    if (!CloneMap_.count(OriginalNode)) {
      return nullptr;
    }
    return CloneMap_[OriginalNode];
  }
  std::vector<CHIPGraphNode *> getLeafNodes();
  std::vector<CHIPGraphNode *> getRootNodes();
  CHIPGraphNode *getClonedNodeFromOriginal(CHIPGraphNode *OriginalNode) {
    if (!CloneMap_.count(OriginalNode)) {
      CHIPERR_LOG_AND_THROW("Failed to find the node in clone", hipErrorTbd);
    } else {
      return CloneMap_[OriginalNode];
    }
  }

  std::vector<CHIPGraphNode *> getNodes() const { return Nodes_; }

  std::vector<std::pair<CHIPGraphNode *, CHIPGraphNode *>> getEdges() {
    std::set<std::pair<CHIPGraphNode *, CHIPGraphNode *>> Edges;
    for (auto Node : Nodes_) {
      for (auto Dep : Node->getDependenciesVec()) {
        auto FromToPair =
            std::pair<CHIPGraphNode *, CHIPGraphNode *>(Node, Dep);
        Edges.insert(FromToPair);
      }
    }

    return std::vector<std::pair<CHIPGraphNode *, CHIPGraphNode *>>(
        Edges.begin(), Edges.end());
  };

  /**
   * @brief Verify/Find node in a graph.
   * HIP API gives const handles to nodes. We can use this function to
   * verify that the node exists in this graph and return the non-const handle.
   *
   * @param Node the node to find in this graph
   * @return CHIPGraphNode*  the non-const handle of this found node.
   */
  CHIPGraphNode *findNode(CHIPGraphNode *Node) {
    auto FoundNode = std::find(Nodes_.begin(), Nodes_.end(), Node);
    if (FoundNode != Nodes_.end()) {
      return *FoundNode;
    } else {
      return nullptr;
    }
  }
};

class CHIPGraphExec {
protected:
  bool Pruned_ = false;
  CHIPGraph *Graph_;
  // each element in this queue represents represents a sequence of nodes that
  // can be submitted to one or more queues
  std::queue<std::set<CHIPGraphNode *>> ExecQueues_;
  std::set<CHIPGraphNode *> findRootNodesSet_() {
    RootNodes_.clear();

    std::vector<CHIPGraphNode *> Nodes = Graph_->getNodes();
    std::set<CHIPGraphNode *> RootNodes;

    auto NodeIter = Nodes.begin();
    while (NodeIter != Nodes.end()) {
      if ((*NodeIter)->getDependenciesSet().size() == 0) {
        RootNodes.insert(*NodeIter);
        RootNodes_.push_back(*NodeIter);
        Nodes.erase(NodeIter);
      } else {
        NodeIter++;
      }
    }

    Pruned_ = true;
    return RootNodes;
  }

  std::vector<CHIPGraphNode *> RootNodes_;

public:
  CHIPGraph *getGraph() const { return Graph_; }
  CHIPGraphExec(CHIPGraph *Graph) {
    // TODO Graphs CHIPExecItem copy constructor failing to copy
    Graph_ = new CHIPGraph(*Graph); // use copy constructor
    // Graph_ = Graph;
  }
  // TODO Graphs - destructor
  void launch(CHIPQueue *Queue);
  void compile();
  /**
   * @brief remove unnecessary dependencies
   * for leaf node in graph:
   *   for each dependency in node:
   *     1. traverse from node to root, constructing a vector of nodes visited
   *   for each traversal vector:
   *     1. traverse from node to root, constructing a vector of nodes visited
   *
   *
   */
  void pruneGraph();
};

#endif // include guard