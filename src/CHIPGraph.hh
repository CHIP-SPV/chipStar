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
class Kernel;
class Event;
class ExecItem;
class Graph;
class GraphNode;
} // namespace chipstar

namespace chipstar {

class GraphNode : public hipGraphNode {
protected:
  hipGraphNodeType Type_;
  // nodes which depend on this node
  std::vector<GraphNode *> Dependendants_;
  // nodes on which this node depends
  std::vector<GraphNode *> Dependencies_;
  /**
   * @brief Destroy the GraphNode object
   * Hidden virtual destructor. Should only be called through derived classes.
   */
  virtual ~GraphNode() {
    Dependendants_.clear();
    Dependencies_.clear();
  }

  GraphNode(hipGraphNodeType Type) : Type_(Type) {}

  void checkDependencies(size_t numDependencies, GraphNode **pDependencies);

public:
  std::string Msg; // TODO Graphs cleanup
  GraphNode(const GraphNode &Other)
      : Type_(Other.Type_), Dependendants_(Other.Dependendants_),
        Dependencies_(Other.Dependencies_), Msg(Other.Msg) {}

  hipGraphNodeType getType() { return Type_; }
  virtual GraphNode *clone() const = 0;

  /**
   * @brief Depth-first search of the graph.
   * For each node, the function is called recursively on all of its
   * dependencies. Returns all possible paths from this node to root node.
   *
   * @param CurrPath space for the current path
   * @param Paths space for the paths
   */
  void DFS(std::vector<GraphNode *> CurrPath,
           std::vector<std::vector<GraphNode *>> &Paths);

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
  void addDependant(GraphNode *TheNode) {
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
  void addDependants(std::vector<GraphNode *> Nodes) {
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
  void addDependency(GraphNode *TheNode) {
    if (TheNode == nullptr)
      CHIPERR_LOG_AND_THROW("addDependency called with nullptr",
                            hipErrorInvalidValue);
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
  void removeDependency(GraphNode *TheNode) {
    if (TheNode == nullptr) {
      CHIPERR_LOG_AND_THROW("removeDependency called with nullptr",
                            hipErrorInvalidValue);
    }
    logDebug("{} removeDependency() <{} depends on {}>", (void *)this, Msg,
             TheNode->Msg);
    auto FoundNode =
        std::find(Dependencies_.begin(), Dependencies_.end(), TheNode);
    if (FoundNode != Dependencies_.end()) {
      Dependencies_.erase(FoundNode);
    } else {
      CHIPERR_LOG_AND_THROW("Failed to find", hipErrorInvalidValue);
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
  void addDependencies(GraphNode **Dependencies, size_t Count) {
    checkDependencies(Count, Dependencies);
    for (size_t i = 0; i < Count; i++) {
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
  void addDependencies(std::vector<GraphNode *> Dependencies) {
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
  void removeDependencies(GraphNode **Dependencies, size_t Count) {
    checkDependencies(Count, Dependencies);
    for (size_t i = 0; i < Count; i++) {
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
  void updateDependencies(std::map<GraphNode *, GraphNode *> &CloneMap) {
    std::vector<GraphNode *> NewDeps;
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
  void updateDependants(std::map<GraphNode *, GraphNode *> CloneMap) {
    std::vector<GraphNode *> NewDeps;
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
   * @return std::vector<GraphNode *>
   */
  std::vector<GraphNode *> getDependencies() const { return Dependencies_; }

  /**
   * @brief Get the Dependants object
   * nodes on which this node depends
   *
   * @return std::vector<GraphNode *>
   */
  std::vector<GraphNode *> getDependants() const { return Dependendants_; }
};

class GraphNodeKernel : public GraphNode {
private:
  hipKernelNodeParams Params_;
  chipstar::ExecItem *ExecItem_;
  chipstar::Kernel *Kernel_;

public:
  GraphNodeKernel(const GraphNodeKernel &Other);

  GraphNodeKernel(const hipKernelNodeParams *TheParams);

  GraphNodeKernel(const void *HostFunction, dim3 GridDim, dim3 BlockDim,
                  void **Args, size_t SharedMem);

  virtual ~GraphNodeKernel() override;
  virtual void execute(chipstar::Queue *Queue) const override;

  hipKernelNodeParams getParams() const { return Params_; }

  /// the Kernel arguments have to be setup either just before launch (when
  /// using the execute() path), or if using the CHIPGraphNative then
  /// just before calling their graph construction APIs.
  ///
  /// This is because Kernels in both LevelZero and OpenCL are stateful,
  /// and users can add multiple nodes with the same kernel into a Graph.
  /// Setting up arguments in GraphNodeKernel ctor would then
  /// lead to all nodes using the same (those set up last) arguments.
  void setupKernelArgs() const;
  chipstar::Kernel *getKernel() const { return Kernel_; }

  void setParams(const hipKernelNodeParams Params) {
    // dont allow changing kernel, needs refactoring
    CHIPASSERT(Params.func == Params_.func);
    Params_ = Params;
  }
  /**
   * @brief Createa a copy of this node
   * Must copy over all the arguments
   * Must copy over all the dependencies.
   * Copying over the dependencies is important because CHIPGraph::clone() uses
   * them to remap onto new nodes
   *
   * @return GraphNode*
   */
  virtual GraphNode *clone() const override;
};

class GraphNodeMemcpy : public GraphNode {
private:
  hipMemcpy3DParms Params_;

  void *Dst_;
  const void *Src_;
  size_t Count_;
  hipMemcpyKind Kind_;

public:
  GraphNodeMemcpy(const GraphNodeMemcpy &Other)
      : GraphNode(Other), Params_(Other.Params_), Dst_(Other.Dst_),
        Src_(Other.Src_), Count_(Other.Count_), Kind_(Other.Kind_) {}

  GraphNodeMemcpy(hipMemcpy3DParms Params)
      : GraphNode(hipGraphNodeTypeMemcpy), Params_(Params), Dst_(nullptr),
        Src_(nullptr), Count_(0), Kind_(hipMemcpyKind::hipMemcpyDefault) {}
  GraphNodeMemcpy(const hipMemcpy3DParms *Params)
      : GraphNode(hipGraphNodeTypeMemcpy) {
    setParams(Params);
  }

  // 1D MemCpy
  GraphNodeMemcpy(void *Dst, const void *Src, size_t Count, hipMemcpyKind Kind)
      : GraphNode(hipGraphNodeTypeMemcpy), Dst_(Dst), Src_(Src), Count_(Count),
        Kind_(Kind) {}

  virtual ~GraphNodeMemcpy() override {}

  hipMemcpy3DParms getParams() { return Params_; }

  // 1D MemCpy
  void setParams(void *Dst, const void *Src, size_t Count, hipMemcpyKind Kind) {
    Dst_ = Dst;
    Src_ = Src;
    Count_ = Count;
    Kind_ = Kind;
  }

  void getParams(void *&Dst, const void *&Src, size_t &Count,
                 hipMemcpyKind &Kind) {
    Dst = Dst_;
    Src = Src_;
    Count = Count_;
    Kind = Kind_;
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

  virtual GraphNode *clone() const override {
    auto NewNode = new GraphNodeMemcpy(*this);
    return NewNode;
  }
};

class GraphNodeMemset : public GraphNode {
private:
  hipMemsetParams Params_;

public:
  GraphNodeMemset(const GraphNodeMemset &Other)
      : GraphNode(Other), Params_(Other.Params_) {}

  GraphNodeMemset(const hipMemsetParams Params)
      : GraphNode(hipGraphNodeTypeMemset), Params_(Params) {}

  GraphNodeMemset(const hipMemsetParams *Params)
      : GraphNode(hipGraphNodeTypeMemset), Params_(*Params) {}

  virtual ~GraphNodeMemset() override {}

  hipMemsetParams getParams() { return Params_; }
  void setParams(const hipMemsetParams *Params) { Params_ = *Params; }

  virtual void execute(chipstar::Queue *Queue) const override;
  virtual GraphNode *clone() const override {
    auto NewNode = new GraphNodeMemset(*this);
    return NewNode;
  }
};

class GraphNodeHost : public GraphNode {
private:
  hipHostNodeParams Params_;

public:
  GraphNodeHost(const GraphNodeHost &Other)
      : GraphNode(Other), Params_(Other.Params_) {}

  GraphNodeHost(const hipHostNodeParams *Params)
      : GraphNode(hipGraphNodeTypeHost), Params_(*Params) {}

  virtual ~GraphNodeHost() override {}

  virtual void execute(chipstar::Queue *Queue) const override;

  virtual GraphNode *clone() const override {
    auto NewNode = new GraphNodeHost(*this);
    return NewNode;
  }

  void setParams(const hipHostNodeParams *Params) { Params_ = *Params; }

  hipHostNodeParams getParams() { return Params_; }
};

class GraphNodeGraph : public GraphNode {
private:
  Graph *SubGraph_;

public:
  GraphNodeGraph(Graph *Graph)
      : GraphNode(hipGraphNodeTypeGraph), SubGraph_(Graph) {}

  GraphNodeGraph(const GraphNodeGraph &Other)
      : GraphNode(Other), SubGraph_(Other.SubGraph_) {}

  virtual ~GraphNodeGraph() override {}

  virtual void execute(chipstar::Queue *Queue) const override {
    CHIPERR_LOG_AND_THROW("Attemped to execute GraphNode", hipErrorTbd);
  }
  virtual GraphNode *clone() const override {
    auto NewNode = new GraphNodeGraph(*this);
    return NewNode;
  }

  void setGraph(Graph *Graph) { SubGraph_ = Graph; }

  Graph *getGraph() { return SubGraph_; }
};

class GraphNodeEmpty : public GraphNode {
public:
  GraphNodeEmpty(const GraphNodeEmpty &Other) : GraphNode(Other) {}

  GraphNodeEmpty() : GraphNode(hipGraphNodeTypeEmpty) {}

  virtual ~GraphNodeEmpty() override {}

  virtual void execute(chipstar::Queue *Queue) const override {
    logDebug("Executing empty node");
  }

  virtual GraphNode *clone() const override {
    auto NewNode = new GraphNodeEmpty(*this);
    return NewNode;
  }
};

class GraphNodeWaitEvent : public GraphNode {
private:
  chipstar::Event *Event_;

public:
  GraphNodeWaitEvent(chipstar::Event *Event)
      : GraphNode(hipGraphNodeTypeWaitEvent), Event_(Event) {}

  GraphNodeWaitEvent(const GraphNodeWaitEvent &Other)
      : GraphNode(Other), Event_(Other.Event_) {}

  virtual ~GraphNodeWaitEvent() override {}

  virtual void execute(chipstar::Queue *Queue) const override;

  virtual GraphNode *clone() const override {
    auto NewNode = new GraphNodeWaitEvent(*this);
    return NewNode;
  }

  chipstar::Event *getEvent() { return Event_; }
  void setEvent(chipstar::Event *Event) { Event_ = Event; }
};

class GraphNodeEventRecord : public GraphNode {
private:
  chipstar::Event *Event_;

public:
  GraphNodeEventRecord(chipstar::Event *Event)
      : GraphNode(hipGraphNodeTypeEventRecord), Event_(Event){};

  GraphNodeEventRecord(const GraphNodeEventRecord &Other)
      : GraphNode(Other), Event_(Other.Event_) {}

  virtual ~GraphNodeEventRecord() override {}

  virtual void execute(chipstar::Queue *Queue) const override;

  virtual GraphNode *clone() const override {
    auto NewNode = new GraphNodeEventRecord(*this);
    return NewNode;
  }

  void setEvent(chipstar::Event *NewEvent) { Event_ = NewEvent; }

  chipstar::Event *getEvent() { return Event_; }
};

class GraphNodeMemcpyFromSymbol : public GraphNode {
private:
  void *Dst_;
  void *Symbol_;
  size_t SizeBytes_;
  size_t Offset_;
  hipMemcpyKind Kind_;

public:
  GraphNodeMemcpyFromSymbol(void *Dst, const void *Symbol, size_t SizeBytes,
                            size_t Offset, hipMemcpyKind Kind)
      : GraphNode(hipGraphNodeTypeMemcpyFromSymbol), Dst_(Dst),
        Symbol_(const_cast<void *>(Symbol)), SizeBytes_(SizeBytes),
        Offset_(Offset), Kind_(Kind) {}

  GraphNodeMemcpyFromSymbol(const GraphNodeMemcpyFromSymbol &Other)
      : GraphNode(Other), Dst_(Other.Dst_), Symbol_(Other.Symbol_),
        SizeBytes_(Other.SizeBytes_), Offset_(Other.Offset_),
        Kind_(Other.Kind_) {}

  virtual ~GraphNodeMemcpyFromSymbol() override {}

  virtual void execute(chipstar::Queue *Queue) const override;

  void setParams(void *Dst, const void *Symbol, size_t SizeBytes, size_t Offset,
                 hipMemcpyKind Kind) {
    Dst_ = Dst;
    Symbol_ = const_cast<void *>(Symbol);
    SizeBytes_ = SizeBytes;
    Offset_ = Offset;
    Kind_ = Kind;
  }

  void getParams(void *&Dst, const void *&Symbol, size_t &SizeBytes,
                 size_t &Offset, hipMemcpyKind &Kind) {
    Dst = Dst_;
    Symbol = Symbol_;
    SizeBytes = SizeBytes_;
    Offset = Offset_;
    Kind = Kind_;
  }

  virtual GraphNode *clone() const override {
    auto NewNode = new GraphNodeMemcpyFromSymbol(*this);
    return NewNode;
  }
};

class GraphNodeMemcpyToSymbol : public GraphNode {
private:
  void *Src_;
  void *Symbol_;
  size_t SizeBytes_;
  size_t Offset_;
  hipMemcpyKind Kind_;

public:
  GraphNodeMemcpyToSymbol(void *Src, const void *Symbol, size_t SizeBytes,
                          size_t Offset, hipMemcpyKind Kind)
      : GraphNode(hipGraphNodeTypeMemcpyToSymbol), Src_(Src),
        Symbol_(const_cast<void *>(Symbol)), SizeBytes_(SizeBytes),
        Offset_(Offset), Kind_(Kind) {}

  GraphNodeMemcpyToSymbol(const GraphNodeMemcpyToSymbol &Other)
      : GraphNode(Other), Src_(Other.Src_), Symbol_(Other.Symbol_),
        SizeBytes_(Other.SizeBytes_), Offset_(Other.Offset_),
        Kind_(Other.Kind_) {}

  virtual ~GraphNodeMemcpyToSymbol() override {}

  virtual void execute(chipstar::Queue *Queue) const override;

  virtual GraphNode *clone() const override {
    auto NewNode = new GraphNodeMemcpyToSymbol(*this);
    return NewNode;
  }

  void setParams(void *Src, const void *Symbol, size_t SizeBytes, size_t Offset,
                 hipMemcpyKind Kind) {
    Src_ = Src;
    Symbol_ = const_cast<void *>(Symbol);
    SizeBytes_ = SizeBytes;
    Offset_ = Offset;
    Kind_ = Kind;
  }

  void getParams(void *&Src, const void *&Symbol, size_t &SizeBytes,
                 size_t &Offset, hipMemcpyKind &Kind) {
    Src = Src_;
    Symbol = Symbol_;
    SizeBytes = SizeBytes_;
    Offset = Offset_;
    Kind = Kind_;
  }
};

class Graph : public ihipGraph {
protected:
  std::vector<GraphNode *> Nodes_;
  // Map the pointers Original -> Clone
  std::map<GraphNode *, GraphNode *> CloneMap_;

public:
  Graph(const Graph &OriginalGraph);
  Graph() {}
  void addNode(GraphNode *TheNode);
  void removeNode(GraphNode *TheNode);
  /**
   * @brief Lookup a cloned(instantiated) node using a pointer to the original
   * node
   *
   * @param OriginalNode pointer to the node which was present in CHIPGraph at
   * the time of instantiation of CHIPGraph to CHIPGraphExec
   * @return GraphNode* pointer to the resulting node in CHIPGraphExec which
   * corresponds to the original node
   */
  GraphNode *nodeLookup(GraphNode *OriginalNode) {
    if (!CloneMap_.count(OriginalNode)) {
      return nullptr;
    }
    return CloneMap_[OriginalNode];
  }
  std::vector<GraphNode *> getLeafNodes();
  std::vector<GraphNode *> getRootNodes();
  GraphNode *getClonedNodeFromOriginal(GraphNode *OriginalNode) {
    if (!CloneMap_.count(OriginalNode)) {
      CHIPERR_LOG_AND_THROW("Failed to find the node in clone",
                            hipErrorInvalidValue);
    } else {
      return CloneMap_[OriginalNode];
    }
  }

  std::vector<GraphNode *> &getNodes() { return Nodes_; }

  std::vector<std::pair<GraphNode *, GraphNode *>> getEdges() {
    std::set<std::pair<GraphNode *, GraphNode *>> Edges;
    for (auto Node : Nodes_) {
      for (auto Dep : Node->getDependencies()) {
        auto FromToPair = std::pair<GraphNode *, GraphNode *>(Node, Dep);
        Edges.insert(FromToPair);
      }
    }

    return std::vector<std::pair<GraphNode *, GraphNode *>>(Edges.begin(),
                                                            Edges.end());
  };

  /**
   * @brief Verify/Find node in a graph.
   * HIP API gives const handles to nodes. We can use this function to
   * verify that the node exists in this graph and return the non-const handle.
   *
   * @param Node the node to find in this graph
   * @return GraphNode*  the non-const handle of this found node.
   */
  GraphNode *findNode(GraphNode *Node) {
    auto FoundNode = std::find(Nodes_.begin(), Nodes_.end(), Node);
    if (FoundNode != Nodes_.end()) {
      return *FoundNode;
    } else {
      return nullptr;
    }
  }
};

class GraphNative {
protected:
  bool Finalized;

public:
  GraphNative() : Finalized(false){};
  virtual ~GraphNative() {}
  bool isFinalized() { return Finalized; }
  virtual bool finalize() { return false; }
  virtual bool addNode(GraphNode *NewNode) { return false; }
};

class GraphExec : public hipGraphExec {
protected:
  Graph *OriginalGraph_;
  Graph CompiledGraph_;

  std::unique_ptr<GraphNative> NativeGraph;

  /**
   * @brief each element in this queue represents represents a sequence of nodes
   * that can be submitted to one or more queues
   *
   */
  std::queue<std::set<GraphNode *>> ExecQueues_;

  /**
   * @brief For every GraphNodeGraph in CompiledGraph_, replace this node
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

public:
  GraphExec(Graph *Graph)
      : OriginalGraph_(Graph), /* Copy the pointer to the original graph */
        CompiledGraph_(*Graph) /* invoke the copy constructor to make
                                             a clone of the graph */
  {}

  ~GraphExec() {}

  void launch(chipstar::Queue *Queue);
  bool tryLaunchNative(chipstar::Queue *Queue);

  Graph *getOriginalGraphPtr() const { return OriginalGraph_; }
};

} // namespace chipstar

#endif // include guard
