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

namespace chipstar {
class Queue;
class Event;
class ExecItem;
} // namespace chipstar

class CHIPGraph;

class CHIPGraphNode : public hipGraphNode {
protected:
  hipGraphNodeType Type_;
  // nodes which depend on this node
  std::vector<CHIPGraphNode *> Dependendants_;
  // nodes on which this node depends
  std::vector<CHIPGraphNode *> Dependencies_;
  /**
   * @brief Destroy the CHIPGraphNode object
   * Hidden virtual destructor. Should only be called through derived classes.
   */
  virtual ~CHIPGraphNode() {
    Dependendants_.clear();
    Dependencies_.clear();
  }

  CHIPGraphNode(hipGraphNodeType Type) : Type_(Type) {}

public:
  std::string Msg; // TODO Graphs cleanup
  CHIPGraphNode(const CHIPGraphNode &Other)
      : Type_(Other.Type_), Dependendants_(Other.Dependendants_),
        Dependencies_(Other.Dependencies_), Msg(Other.Msg) {}

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

  /**
   * @brief Pure virtual method to be overriden by derived classes. This method
   * gets called during graph execution.
   *
   * @param Queue Queue in which to execute this node
   */
  virtual void execute(chipstar::Queue *Queue) const = 0;

  /**
   * @brief Add a dependant to a node.
   *
   * Visualizing the graph, add an edge going up.
   *
   * @param TheNode
   */
  void addDependant(CHIPGraphNode *TheNode) {
    logDebug("{} addDependant() <{} depends on {}>", (void *)this, TheNode->Msg,
             Msg);
    Dependendants_.push_back(TheNode);
  }

  /**
   * @brief Add a dependant to a node.
   *
   * Visualizing the graph, add an edge going up.
   *
   * @param TheNode
   */
  void addDependants(std::vector<CHIPGraphNode *> Nodes) {
    for (auto Node : Nodes) {
      addDependant(Node);
    }
  }

  /**
   * @brief Add a dependency to a node.
   *
   * Visualizing the graph, add an edge going down.
   *
   * @param TheNode
   */
  void addDependency(CHIPGraphNode *TheNode) {
    logDebug("{} addDependency() <{} depends on {}>", (void *)this, Msg,
             TheNode->Msg);
    Dependencies_.push_back(TheNode);
    TheNode->addDependant(this);
  }

  /**
   * @brief  Remove a dependency from a node.
   *
   * Visualizing the graph, remove an edge going down.
   *
   * @param TheNode
   */
  void removeDependency(CHIPGraphNode *TheNode) {
    logDebug("{} removeDependency() <{} depends on {}>", (void *)this, Msg,
             TheNode->Msg);
    auto FoundNode =
        std::find(Dependencies_.begin(), Dependencies_.end(), TheNode);
    if (FoundNode != Dependencies_.end()) {
      Dependencies_.erase(FoundNode);
    } else {
      CHIPERR_LOG_AND_THROW("Failed to find", hipErrorTbd);
    }
  }

  /**
   * @brief  Add a dependency from a node.
   *
   * Visualizing the graph, add an edge going down.
   *
   * @param Dependencies
   * @param Count
   */
  void addDependencies(CHIPGraphNode **Dependencies, int Count) {
    for (int i = 0; i < Count; i++) {
      addDependency(Dependencies[i]);
    }
  }

  /**
   * @brief  Add a dependency from a node.
   *
   * Visualizing the graph, add an edge going down.
   *
   * @param Dependencies
   */
  void addDependencies(std::vector<CHIPGraphNode *> Dependencies) {
    for (auto Node : Dependencies) {
      addDependency(Node);
    }
  }

  /**
   * @brief  Remove multiple dependencies from a node.
   *
   * Visualizing the graph, remove edges going down.
   *
   * @param TheNode
   */
  void removeDependencies(CHIPGraphNode **Dependencies, int Count) {
    for (int i = 0; i < Count; i++) {
      removeDependency(Dependencies[i]);
    }
  }

  /**
   * @brief Remap dependencies of this graph.
   *
   * When a graph gets cloned, all nodes are cloned from this original graph.
   * After the clone, these nodes are identical which includes the fact that all
   * the dependencies(edges) are poiting to nodes in the original graph. Given a
   * map, remap these dependencies to nodes in this graph instead. CloneMap gets
   * generated when CHIPGraph::clone() is executed.
   *
   * @param CloneMap  the map containing relationships of which original node
   * does each cloned node correspond to.
   */
  void updateDependencies(std::map<CHIPGraphNode *, CHIPGraphNode *> CloneMap) {
    std::vector<CHIPGraphNode *> NewDeps;
    for (auto Dep : Dependencies_) {
      auto ClonedDep = CloneMap[Dep];
      logDebug("{} {} Replacing dependency {} with {}", (void *)this, this->Msg,
               (void *)Dep, (void *)ClonedDep);
      NewDeps.push_back(ClonedDep);
    }
    Dependencies_.clear();
    Dependencies_ = NewDeps;
    return;
  }

  /**
   * @brief Remap dependants of this graph.
   *
   * When a graph gets cloned, all nodes are cloned from this original graph.
   * After the clone, these nodes are identical which includes the fact that all
   * the dependencies(edges) are poiting to nodes in the original graph. Given a
   * map, remap these dependants to nodes in this graph instead. CloneMap gets
   * generated when CHIPGraph::clone() is executed.
   *
   * @param CloneMap  the map containing relationships of which original node
   * does each cloned node correspond to.
   */
  void updateDependants(std::map<CHIPGraphNode *, CHIPGraphNode *> CloneMap) {
    std::vector<CHIPGraphNode *> NewDeps;
    for (auto Dep : Dependendants_) {
      auto ClonedDep = CloneMap[Dep];
      logDebug("{} {} Replacing dependant {} with {}", (void *)this, this->Msg,
               (void *)Dep, (void *)ClonedDep);
      NewDeps.push_back(ClonedDep);
    }
    Dependendants_.clear();
    Dependendants_ = NewDeps;
    return;
  }

  /**
   * @brief Get the Dependencies object
   *  nodes which depend on this node
   *
   * @return std::vector<CHIPGraphNode *>
   */
  std::vector<CHIPGraphNode *> getDependencies() const { return Dependencies_; }

  /**
   * @brief Get the Dependants object
   * nodes on which this node depends
   *
   * @return std::vector<CHIPGraphNode *>
   */
  std::vector<CHIPGraphNode *> getDependants() const { return Dependendants_; }
};

class CHIPGraphNodeKernel : public CHIPGraphNode {
private:
  /// A block holding the bytes of the kernel arguments.
  std::vector<char> ArgData_;

  /// pointer to start of the kernel argument data for each kernel argument.
  std::vector<void *> ArgList_;

  hipKernelNodeParams Params_;
  chipstar::ExecItem *ExecItem_;

public:
  CHIPGraphNodeKernel(const CHIPGraphNodeKernel &Other);

  CHIPGraphNodeKernel(const hipKernelNodeParams *TheParams);

  CHIPGraphNodeKernel(const void *HostFunction, dim3 GridDim, dim3 BlockDim,
                      void **Args, size_t SharedMem);

  virtual ~CHIPGraphNodeKernel() override {}

  virtual void execute(chipstar::Queue *Queue) const override;

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
};

class CHIPGraphNodeMemcpy : public CHIPGraphNode {
private:
  hipMemcpy3DParms Params_;

  void *Dst_;
  const void *Src_;
  size_t Count_;
  hipMemcpyKind Kind_;

public:
  CHIPGraphNodeMemcpy(const CHIPGraphNodeMemcpy &Other)
      : CHIPGraphNode(Other), Params_(Other.Params_), Dst_(Other.Dst_),
        Src_(Other.Src_), Count_(Other.Count_), Kind_(Other.Kind_) {}

  CHIPGraphNodeMemcpy(hipMemcpy3DParms Params)
      : CHIPGraphNode(hipGraphNodeTypeMemcpy), Params_(Params) {}
  CHIPGraphNodeMemcpy(const hipMemcpy3DParms *Params)
      : CHIPGraphNode(hipGraphNodeTypeMemcpy) {
    setParams(Params);
  }

  // 1D MemCpy
  CHIPGraphNodeMemcpy(void *Dst, const void *Src, size_t Count,
                      hipMemcpyKind Kind)
      : CHIPGraphNode(hipGraphNodeTypeMemcpy), Dst_(Dst), Src_(Src),
        Count_(Count), Kind_(Kind) {}

  virtual ~CHIPGraphNodeMemcpy() override {}

  hipMemcpy3DParms getParams() { return Params_; }

  // 1D MemCpy
  void setParams(void *Dst, const void *Src, size_t Count, hipMemcpyKind Kind) {
    Dst_ = Dst;
    Src_ = Src;
    Count_ = Count;
    Kind_ = Kind;
  }

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

  virtual void execute(chipstar::Queue *Queue) const override;

  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeMemcpy(*this);
    return NewNode;
  }
};

class CHIPGraphNodeMemset : public CHIPGraphNode {
private:
  hipMemsetParams Params_;

public:
  CHIPGraphNodeMemset(const CHIPGraphNodeMemset &Other)
      : CHIPGraphNode(Other), Params_(Other.Params_) {}

  CHIPGraphNodeMemset(const hipMemsetParams Params)
      : CHIPGraphNode(hipGraphNodeTypeMemset), Params_(Params) {}

  CHIPGraphNodeMemset(const hipMemsetParams *Params)
      : CHIPGraphNode(hipGraphNodeTypeMemset), Params_(*Params) {}

  virtual ~CHIPGraphNodeMemset() override {}

  hipMemsetParams getParams() { return Params_; }
  void setParams(const hipMemsetParams *Params) { Params_ = *Params; }

  virtual void execute(chipstar::Queue *Queue) const override;
  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeMemset(*this);
    return NewNode;
  }
};

class CHIPGraphNodeHost : public CHIPGraphNode {
private:
  hipHostNodeParams Params_;

public:
  CHIPGraphNodeHost(const CHIPGraphNodeHost &Other)
      : CHIPGraphNode(Other), Params_(Other.Params_) {}

  CHIPGraphNodeHost(const hipHostNodeParams *Params)
      : CHIPGraphNode(hipGraphNodeTypeHost), Params_(*Params) {}

  virtual ~CHIPGraphNodeHost() override {}

  virtual void execute(chipstar::Queue *Queue) const override;

  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeHost(*this);
    return NewNode;
  }

  void setParams(const hipHostNodeParams *Params) { Params_ = *Params; }

  hipHostNodeParams getParams() { return Params_; }
};

class CHIPGraphNodeGraph : public CHIPGraphNode {
private:
  CHIPGraph *SubGraph_;

public:
  CHIPGraphNodeGraph(CHIPGraph *Graph)
      : CHIPGraphNode(hipGraphNodeTypeGraph), SubGraph_(Graph) {}

  CHIPGraphNodeGraph(const CHIPGraphNodeGraph &Other)
      : CHIPGraphNode(Other), SubGraph_(Other.SubGraph_) {}

  virtual ~CHIPGraphNodeGraph() override {}

  virtual void execute(chipstar::Queue *Queue) const override {
    CHIPERR_LOG_AND_THROW("Attemped to execute GraphNode", hipErrorTbd);
  }
  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeGraph(*this);
    return NewNode;
  }

  void setGraph(CHIPGraph *Graph) { SubGraph_ = Graph; }

  CHIPGraph *getGraph() { return SubGraph_; }
};

class CHIPGraphNodeEmpty : public CHIPGraphNode {
public:
  CHIPGraphNodeEmpty(const CHIPGraphNodeEmpty &Other) : CHIPGraphNode(Other) {}

  CHIPGraphNodeEmpty() : CHIPGraphNode(hipGraphNodeTypeEmpty) {}

  virtual ~CHIPGraphNodeEmpty() override {}

  virtual void execute(chipstar::Queue *Queue) const override {
    logDebug("Executing empty node");
  }

  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeEmpty(*this);
    return NewNode;
  }
};

class CHIPGraphNodeWaitEvent : public CHIPGraphNode {
private:
  chipstar::Event *Event_;

public:
  CHIPGraphNodeWaitEvent(chipstar::Event *Event)
      : CHIPGraphNode(hipGraphNodeTypeWaitEvent), Event_(Event) {}

  CHIPGraphNodeWaitEvent(const CHIPGraphNodeWaitEvent &Other)
      : CHIPGraphNode(Other), Event_(Other.Event_) {}

  virtual ~CHIPGraphNodeWaitEvent() override {}

  virtual void execute(chipstar::Queue *Queue) const override;

  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeWaitEvent(*this);
    return NewNode;
  }

  chipstar::Event *getEvent() { return Event_; }
  void setEvent(chipstar::Event *Event) { Event_ = Event; }
};

class CHIPGraphNodeEventRecord : public CHIPGraphNode {
private:
  chipstar::Event *Event_;

public:
  CHIPGraphNodeEventRecord(chipstar::Event *Event)
      : CHIPGraphNode(hipGraphNodeTypeEventRecord), Event_(Event){};

  CHIPGraphNodeEventRecord(const CHIPGraphNodeEventRecord &Other)
      : CHIPGraphNode(Other), Event_(Other.Event_) {}

  virtual ~CHIPGraphNodeEventRecord() override {}

  virtual void execute(chipstar::Queue *Queue) const override;

  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeEventRecord(*this);
    return NewNode;
  }

  void setEvent(chipstar::Event *NewEvent) { Event_ = NewEvent; }

  chipstar::Event *getEvent() { return Event_; }
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
      : CHIPGraphNode(hipGraphNodeTypeMemcpyFromSymbol), Dst_(Dst),
        Symbol_(const_cast<void *>(Symbol)), SizeBytes_(SizeBytes),
        Offset_(Offset), Kind_(Kind) {}

  CHIPGraphNodeMemcpyFromSymbol(const CHIPGraphNodeMemcpyFromSymbol &Other)
      : CHIPGraphNode(Other), Dst_(Other.Dst_), Symbol_(Other.Symbol_),
        SizeBytes_(Other.SizeBytes_), Offset_(Other.Offset_),
        Kind_(Other.Kind_) {}

  virtual ~CHIPGraphNodeMemcpyFromSymbol() override {}

  virtual void execute(chipstar::Queue *Queue) const override;

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
      : CHIPGraphNode(hipGraphNodeTypeMemcpyToSymbol), Src_(Src),
        Symbol_(const_cast<void *>(Symbol)), SizeBytes_(SizeBytes),
        Offset_(Offset), Kind_(Kind) {}

  CHIPGraphNodeMemcpyToSymbol(const CHIPGraphNodeMemcpyToSymbol &Other)
      : CHIPGraphNode(Other), Src_(Other.Src_), Symbol_(Other.Symbol_),
        SizeBytes_(Other.SizeBytes_), Offset_(Other.Offset_),
        Kind_(Other.Kind_) {}

  virtual ~CHIPGraphNodeMemcpyToSymbol() override {}

  virtual void execute(chipstar::Queue *Queue) const override;

  virtual CHIPGraphNode *clone() const override {
    auto NewNode = new CHIPGraphNodeMemcpyToSymbol(*this);
    return NewNode;
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

class CHIPGraph : public ihipGraph {
protected:
  std::vector<CHIPGraphNode *> Nodes_;
  // Map the pointers Original -> Clone
  std::map<CHIPGraphNode *, CHIPGraphNode *> CloneMap_;

public:
  CHIPGraph(const CHIPGraph &OriginalGraph);
  CHIPGraph() {}
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

  std::vector<CHIPGraphNode *> &getNodes() { return Nodes_; }

  std::vector<std::pair<CHIPGraphNode *, CHIPGraphNode *>> getEdges() {
    std::set<std::pair<CHIPGraphNode *, CHIPGraphNode *>> Edges;
    for (auto Node : Nodes_) {
      for (auto Dep : Node->getDependencies()) {
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

class CHIPGraphExec : public hipGraphExec {
protected:
  CHIPGraph *OriginalGraph_;
  CHIPGraph CompiledGraph_;

  /**
   * @brief each element in this queue represents represents a sequence of nodes
   * that can be submitted to one or more queues
   *
   */
  std::queue<std::set<CHIPGraphNode *>> ExecQueues_;

  /**
   * @brief For every CHIPGraphNodeGraph in CompiledGraph_, replace this node
   * with its contents.
   *
   */
  void ExtractSubGraphs_();

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
  void pruneGraph_();

public:
  CHIPGraphExec(CHIPGraph *Graph)
      : OriginalGraph_(Graph), /* Copy the pointer to the original graph */
        CompiledGraph_(CHIPGraph(*Graph)) /* invoke the copy constructor to make
                                             a clone of the graph */
  {}

  ~CHIPGraphExec() {}

  void launch(chipstar::Queue *Queue);

  CHIPGraph *getOriginalGraphPtr() const { return OriginalGraph_; }

  /**
   * @brief Optimize and generate ExecQueues_
   *
   * This method will first call PruneGraph and then generate an executable
   * queue. Executable queue is made up of sets of nodes. All members of the
   * aforementioned set can be executed simultanously in no particular order.
   * @see PruneGraph
   *
   */
  void compile();
};

#endif // include guard
