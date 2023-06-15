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
 * @file CHIPBindings.hh
 * @author Paulius Velesko (pvelesko@pglc.io)
 * @brief Implementations of the HIP API functions using the CHIP interface
 * providing basic functionality such hipMemcpy, host and device function
 * registration, hipLaunchByPtr, etc.
 * These functions operate on base CHIP class pointers allowing for backend
 * selection at runtime and backend-specific implementations are done by
 * inheriting from base CHIP classes and overriding virtual member functions.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef CHIP_BINDINGS_H
#define CHIP_BINDINGS_H
#include <sys/mman.h>
#include <errno.h>
#include <fstream>

#include "CHIPBackend.hh"
#include "CHIPDriver.hh"
#include "CHIPException.hh"
#include "common.hh"
#include "hip/hip_interop.h"
#include "hip/hip_runtime_api.h"
#include "hip_conversions.hh"
#include "macros.hh"
#include "Utils.hh"
#include "SPVRegister.hh"
#include "hipCtx.hh"

#define SVM_ALIGNMENT 128 // TODO Pass as CMAKE Define?

#define GRAPH(x) static_cast<CHIPGraph *>(x)

#define NODE(x) static_cast<CHIPGraphNode *>(x)

#define EXEC(x) static_cast<CHIPGraphExec *>(x)

#define NODES(x) reinterpret_cast<CHIPGraphNode **>(x)

#define DECONST_NODE(x)                                                        \
  static_cast<CHIPGraphNode *>(const_cast<hipGraphNode_t>(x))

#define DECONST_NODES(x)                                                       \
  reinterpret_cast<CHIPGraphNode **>(const_cast<hipGraphNode_t *>(x))

hipError_t hipFreeArray(hipArray *Array);

hipError_t hipDeviceGetP2PAttribute(int *value, hipDeviceP2PAttr attr,
                                    int srcDevice, int dstDevice) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipGetChannelDesc(hipChannelFormatDesc *desc,
                             hipArray_const_t array) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipDeviceGetUuid(hipUUID *uuid, hipDevice_t device) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipExtStreamCreateWithCUMask(hipStream_t *stream,
                                        uint32_t cuMaskSize,
                                        const uint32_t *cuMask) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize,
                                 uint32_t *cuMask) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D *pCopy) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes,
                                      hipPointer_attribute *attributes,
                                      void **data, hipDeviceptr_t ptr) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipMemRangeGetAttributes(void **data, size_t *data_sizes,
                                    hipMemRangeAttribute *attributes,
                                    size_t num_attributes, const void *dev_ptr,
                                    size_t count) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipPointerGetAttribute(void *data, hipPointer_attribute attribute,
                                  hipDeviceptr_t ptr) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D *pCopy, hipStream_t stream) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t *mem_pool, int device) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipArrayDestroy(hipArray *Array) { return hipFreeArray(Array); }

hipError_t hipArray3DCreate(hipArray **array,
                            const HIP_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipMemAllocPitch(hipDeviceptr_t *dptr, size_t *pitch,
                            size_t widthInBytes, size_t height,
                            unsigned int elementSizeBytes) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipDeviceGetMemPool(hipMemPool_t *mem_pool, int device) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipMallocAsync(void **dev_ptr, size_t size, hipStream_t stream) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipFreeAsync(void *dev_ptr, hipStream_t stream) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr,
                                  void *value) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr,
                                  void *value) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool,
                               const hipMemAccessDesc *desc_list,
                               size_t count) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipMemPoolGetAccess(hipMemAccessFlags *flags, hipMemPool_t mem_pool,
                               hipMemLocation *location) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipMemPoolCreate(hipMemPool_t *mem_pool,
                            const hipMemPoolProps *pool_props) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipMallocFromPoolAsync(void **dev_ptr, size_t size,
                                  hipMemPool_t mem_pool, hipStream_t stream) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn,
                             void *userData) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipStreamIsCapturing(hipStream_t stream,
                                hipStreamCaptureStatus *pCaptureStatus) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipStreamGetCaptureInfo(hipStream_t stream,
                                   hipStreamCaptureStatus *pCaptureStatus,
                                   unsigned long long *pId) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream,
                                      hipStreamCaptureStatus *captureStatus_out,
                                      unsigned long long *id_out,
                                      hipGraph_t *graph_out,
                                      const hipGraphNode_t **dependencies_out,
                                      size_t *numDependencies_out) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipUserObjectCreate(hipUserObject_t *object_out, void *ptr,
                               hipHostFn_t destroy,
                               unsigned int initialRefcount,
                               unsigned int flags) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object,
                                    unsigned int count, unsigned int flags) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object,
                                     unsigned int count) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipInit(unsigned int flags) {
  CHIP_TRY
  if (flags)
    RETURN(hipErrorInvalidValue);
  CHIPInitialize();
  RETURN(hipSuccess);
  CHIP_CATCH
};

// Handles device side abort() call by checking the abort flag global
// variable used for signaling the request.
static void handleAbortRequest(CHIPQueue &Q, CHIPModule &M) {
  logTrace("handleAbortRequest()");
  CHIPDeviceVar *Var = M.getGlobalVar("__chipspv_abort_called");

  if (!Var)
    // If the flag is not found, we have removed it in HipAbort pass
    // to denote abort is not called by any kernel in the module. This
    // is used for avoiding kernel launches to read the value to
    // minimize overheads when abort is not used.
    return;

  int32_t AbortFlag = 0;
  hipError_t Err = Q.memCopy(&AbortFlag, Var->getDevAddr(), sizeof(int32_t));
  if (Err != hipSuccess)
    // We know the abort flag exist so what went wrong on the copy?
    CHIPERR_LOG_AND_THROW("Unexpected mem copy failure.", hipErrorTbd);

  if (!AbortFlag)
    return; // Abort was not called.

  // Disable host-side abort behavior for making the unit testing of abort
  // cases easier.
  if (!getenv("CHIP_HOST_IGNORES_DEVICE_ABORT")) {
    // Intel CPU OpenCL doesn't seem flush after the kernel completion.
    std::cout << std::flush;
    abort();
  }

  // Just act like nothing happened. Reset the flag so we let there be more
  // aborts.
  AbortFlag = 0;
  Err = Q.memCopy(Var->getDevAddr(), &AbortFlag, sizeof(int32_t));
  if (Err != hipSuccess)
    // Device->host copy succeeded. What went wrong with host->device copy?
    CHIPERR_LOG_AND_THROW("Unexpected mem copy failure.", hipErrorTbd);

  printf("[ABORT IGNORED]\n");
}

hipError_t hipGraphCreate(hipGraph_t *pGraph, unsigned int flags) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraph *Graph = new CHIPGraph();
  *pGraph = Graph;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphDestroy(hipGraph_t graph) {
  CHIP_TRY
  CHIPInitialize();
  delete graph;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t *from,
                                   const hipGraphNode_t *to,
                                   size_t numDependencies) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphNode *FoundNode = GRAPH(graph)->findNode(NODE(*to));
  if (!FoundNode)
    RETURN(hipErrorInvalidValue);

  FoundNode->addDependencies(DECONST_NODES(from), numDependencies);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphRemoveDependencies(hipGraph_t graph,
                                      const hipGraphNode_t *from,
                                      const hipGraphNode_t *to,
                                      size_t numDependencies) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphNode *FoundNode = GRAPH(graph)->findNode(NODE(*to));
  if (!FoundNode)
    RETURN(hipErrorInvalidValue);

  FoundNode->removeDependencies(DECONST_NODES(from), numDependencies);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t *from,
                            hipGraphNode_t *to, size_t *numEdges) {
  CHIP_TRY
  CHIPInitialize();
  auto Edges = GRAPH(graph)->getEdges();
  if (!to && !from) {
    *numEdges = Edges.size();
    RETURN(hipSuccess);
  }

  for (int i = 0; i < Edges.size(); i++) {
    auto Edge = Edges[i];
    auto FromNode = Edge.first;
    auto ToNode = Edge.second;
    from[i] = FromNode;
    to[i] = ToNode;
  }
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t *nodes,
                            size_t *numNodes) {
  CHIP_TRY
  CHIPInitialize();
  auto Nodes = GRAPH(graph)->getNodes();
  *nodes = *(Nodes.data());
  *numNodes = GRAPH(graph)->getNodes().size();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t *pRootNodes,
                                size_t *pNumRootNodes) {
  CHIP_TRY
  CHIPInitialize();
  auto Nodes = GRAPH(graph)->getRootNodes();
  *pRootNodes = *(Nodes.data());
  *pNumRootNodes = GRAPH(graph)->getNodes().size();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node,
                                       hipGraphNode_t *pDependencies,
                                       size_t *pNumDependencies) {
  CHIP_TRY
  CHIPInitialize();
  auto Deps = NODE(node)->getDependencies();
  *pNumDependencies = Deps.size();
  if (!pDependencies)
    RETURN(hipSuccess);
  for (int i = 0; i < Deps.size(); i++) {
    pDependencies[i] = Deps[i];
  }
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node,
                                         hipGraphNode_t *pDependentNodes,
                                         size_t *pNumDependentNodes) {
  CHIP_TRY
  CHIPInitialize();
  auto Deps = NODE(node)->getDependants();
  *pNumDependentNodes = Deps.size();
  if (!pDependentNodes)
    RETURN(hipSuccess);
  for (int i = 0; i < Deps.size(); i++) {
    pDependentNodes[i] = Deps[i];
  }
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType *pType) {
  CHIP_TRY
  CHIPInitialize();
  *pType = NODE(node)->getType();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
  CHIP_TRY
  CHIPInitialize();
  /**
   * have to resort to these shenanigans to call the proper derived destructor
   */
  auto NodeType = NODE(node)->getType();
  switch (NodeType) {
  case hipGraphNodeTypeKernel:
    delete static_cast<CHIPGraphNodeKernel *>(node);
    break;
  case hipGraphNodeTypeMemcpy:
    delete static_cast<CHIPGraphNodeMemcpy *>(node);
    break;
  case hipGraphNodeTypeMemset:
    delete static_cast<CHIPGraphNodeMemset *>(node);
    break;
  case hipGraphNodeTypeHost:
    delete static_cast<CHIPGraphNodeHost *>(node);
    break;
  case hipGraphNodeTypeGraph:
    delete static_cast<CHIPGraphNodeGraph *>(node);
    break;
  case hipGraphNodeTypeEmpty:
    delete static_cast<CHIPGraphNodeEmpty *>(node);
    break;
  case hipGraphNodeTypeWaitEvent:
    delete static_cast<CHIPGraphNodeWaitEvent *>(node);
    break;
  case hipGraphNodeTypeEventRecord:
    delete static_cast<CHIPGraphNodeEventRecord *>(node);
    break;
  case hipGraphNodeTypeMemcpyFromSymbol:
    delete static_cast<CHIPGraphNodeMemcpyFromSymbol *>(node);
    break;
  case hipGraphNodeTypeMemcpyToSymbol:
    delete static_cast<CHIPGraphNodeMemcpyToSymbol *>(node);
    break;
  default:
    CHIPERR_LOG_AND_THROW("Unknown graph node type", hipErrorTbd);
    break;
  }
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphClone(hipGraph_t *pGraphClone, hipGraph_t originalGraph) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraph *CloneGraph = new CHIPGraph(*GRAPH(originalGraph));
  *pGraphClone = CloneGraph;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphNodeFindInClone(hipGraphNode_t *pNode,
                                   hipGraphNode_t originalNode,
                                   hipGraph_t clonedGraph) {
  CHIP_TRY
  CHIPInitialize();
  auto Node = GRAPH(clonedGraph)->getClonedNodeFromOriginal(NODE(originalNode));
  *pNode = Node;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphInstantiate(hipGraphExec_t *pGraphExec, hipGraph_t graph,
                               hipGraphNode_t *pErrorNode, char *pLogBuffer,
                               size_t bufferSize) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphExec *GraphExec = new CHIPGraphExec(GRAPH(graph));
  *pGraphExec = GraphExec;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t *pGraphExec,
                                        hipGraph_t graph,
                                        unsigned long long flags) {
  CHIP_TRY
  CHIPInitialize();
  // flags not yet defined in HIP API.
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = static_cast<CHIPQueue *>(stream);
  ChipQueue = Backend->findQueue(ChipQueue);
  EXEC(graphExec)->launch(ChipQueue);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
  CHIP_TRY
  CHIPInitialize();
  delete graphExec;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                              hipGraphNode_t *hErrorNode_out,
                              hipGraphExecUpdateResult *updateResult_out) {
  CHIP_TRY
  CHIPInitialize();
  // TODO Graphs - hipGraphExecUpdate
  /**
   * cudaGraphExecUpdate sets updateResult_out to
   cudaGraphExecUpdateErrorTopologyChanged under the following conditions:
    1. The count of nodes directly in hGraphExec and hGraph differ, in which
   case hErrorNode_out is NULL.
    2. A node is deleted in hGraph but not not its pair from hGraphExec, in
   which case hErrorNode_out is NULL.
    3. A node is deleted in hGraphExec but not its pair from hGraph, in which
   case hErrorNode_out is the pairless node from hGraph.
    4. The dependent nodes of a pair differ, in which case hErrorNode_out is the
   node from hGraph.
   */
  auto ExecGraph = EXEC(hGraphExec)->getOriginalGraphPtr();
  // 1.
  if (ExecGraph->getNodes().size() != GRAPH(hGraph)->getNodes().size()) {
    *updateResult_out = hipGraphExecUpdateErrorTopologyChanged;
    *hErrorNode_out = nullptr;
    RETURN(hipErrorGraphExecUpdateFailure);
  }
  // 2.
  if (ExecGraph->getNodes().size() > GRAPH(hGraph)->getNodes().size()) {
    *updateResult_out = hipGraphExecUpdateErrorTopologyChanged;
    *hErrorNode_out = nullptr;
    RETURN(hipErrorGraphExecUpdateFailure);
  }
  // 3.
  for (auto Node : GRAPH(hGraph)->getNodes()) {
    auto NodeFound = ExecGraph->nodeLookup(Node);
    if (!NodeFound) {
      *updateResult_out = hipGraphExecUpdateErrorTopologyChanged;
      *hErrorNode_out = Node;
      RETURN(hipErrorGraphExecUpdateFailure);
    }
  }

  /**
   Update Limitations:
    a) Kernel nodes:
        1. The owning context of the function cannot change.
        2. A node whose function originally did not use CUDA dynamic parallelism
   cannot be updated to a function which uses CDP.
        3. A cooperative node cannot be updated to a non-cooperative node, and
   vice-versa.
        4. If the graph was instantiated with
   cudaGraphInstantiateFlagUseNodePriority, the priority attribute cannot
   change. Equality is checked on the originally requested priority values,
   before they are clamped to the device's supported range.

    b) Memset and memcpy nodes:
        1. The CUDA device(s) to which the operand(s) was allocated/mapped
   cannot change.
        2. The source/destination memory must be allocated from the same
   contexts as the original source/destination memory.
        3. Only 1D memsets can be changed.

    c) Additional memcpy node restrictions:
        1. Changing either the source or destination memory type(i.e.
   CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_ARRAY, etc.) is not supported.
   **/

  /**
   * cudaGraphExecUpdate sets updateResult_out to:
      1. cudaGraphExecUpdateError if passed an invalid value.
      2. cudaGraphExecUpdateErrorTopologyChanged if the graph topology changed
      3. cudaGraphExecUpdateErrorNodeTypeChanged if the type of a node changed,
   in which case hErrorNode_out is set to the node from hGraph.
      4. cudaGraphExecUpdateErrorFunctionChanged if the function of a kernel
   node changed (CUDA driver < 11.2)
      5. cudaGraphExecUpdateErrorUnsupportedFunctionChange if the func field of
   a kernel changed in an unsupported way(see note above), in which case
   hErrorNode_out is set to the node from hGraph
      cudaGraphExecUpdateErrorParametersChanged if any parameters to a node
   changed in a way that is not supported, in which case hErrorNode_out is set
   to the node from hGraph cudaGraphExecUpdateErrorAttributesChanged if any
   attributes of a node changed in a way that is not supported, in which case
   hErrorNode_out is set to the node from hGraph
      cudaGraphExecUpdateErrorNotSupported if something about a node is
   unsupported, like the node's type or configuration, in which case
   hErrorNode_out is set to the node from hGraph
   **/
  for (auto Node : GRAPH(hGraph)->getNodes()) {
    auto NodeFound = ExecGraph->nodeLookup(Node);
    // 3.
    if (Node->getType() != NodeFound->getType()) {
      *updateResult_out = hipGraphExecUpdateErrorNodeTypeChanged;
      *hErrorNode_out = Node;
      RETURN(hipErrorGraphExecUpdateFailure);
    }

    // 4.
    if (Node->getType() == hipGraphNodeType::hipGraphNodeTypeKernel &&
        NodeFound->getType() == hipGraphNodeType::hipGraphNodeTypeKernel) {
      auto NodeCast = static_cast<CHIPGraphNodeKernel *>(Node);
      auto NodeFoundCast = static_cast<CHIPGraphNodeKernel *>(NodeFound);
      if (NodeCast->getParams().func != NodeFoundCast->getParams().func) {
        *updateResult_out = hipGraphExecUpdateErrorFunctionChanged;
        *hErrorNode_out = Node;
        RETURN(hipErrorGraphExecUpdateFailure);
      }
    }
  }

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphAddKernelNode(hipGraphNode_t *pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t *pDependencies,
                                 size_t numDependencies,
                                 const hipKernelNodeParams *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphNodeKernel *Node = new CHIPGraphNodeKernel{pNodeParams};
  Node->addDependencies(DECONST_NODES(pDependencies), numDependencies);
  *pGraphNode = Node;
  GRAPH(graph)->addNode(Node);
  Node->Msg += "Kernel";
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node,
                                       hipKernelNodeParams *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  *pNodeParams = ((CHIPGraphNodeKernel *)node)->getParams();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node,
                                       const hipKernelNodeParams *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  ((CHIPGraphNodeKernel *)node)->setParams(*pNodeParams);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t
hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                const hipKernelNodeParams *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  // Graph obtained from hipGraphExec_t is a clone of the original
  CHIPGraph *Graph = EXEC(hGraphExec)->getOriginalGraphPtr();
  // KernelNode here is a handle to the original

  CHIPGraphNodeKernel *ExecKernelNode = static_cast<CHIPGraphNodeKernel *>(
      GRAPH(Graph)->getClonedNodeFromOriginal(NODE(node)));
  assert(ExecKernelNode);

  ExecKernelNode->setParams(*pNodeParams);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphAddMemcpyNode(hipGraphNode_t *pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t *pDependencies,
                                 size_t numDependencies,
                                 const hipMemcpy3DParms *pCopyParams) {
  CHIP_TRY
  CHIPInitialize();

  // graphs test seems wrong - normally we expect hipErrorInvalidHandle
  // NULLCHECK(graph, pGraphNode, pCopyParams);
  if (!graph || !pGraphNode || !pCopyParams)
    RETURN(hipErrorInvalidValue);
  if (pDependencies == nullptr & numDependencies > 0)
    CHIPERR_LOG_AND_THROW(
        "numDependencies is not 0 while pDependencies is null",
        hipErrorInvalidValue);

  if (!pCopyParams->srcArray && !pCopyParams->srcPtr.ptr)
    CHIPERR_LOG_AND_THROW("all src are null", hipErrorInvalidValue);

  if (!pCopyParams->dstArray && !pCopyParams->dstPtr.ptr)
    CHIPERR_LOG_AND_THROW("all dst are null", hipErrorInvalidValue);

  if ((pCopyParams->dstArray && pCopyParams->srcArray) &&
      ((pCopyParams->dstArray->depth != pCopyParams->srcArray->depth) ||
       (pCopyParams->dstArray->height != pCopyParams->srcArray->height) ||
       (pCopyParams->dstArray->width != pCopyParams->srcArray->width)))
    CHIPERR_LOG_AND_THROW(
        "Passing different element size for hipMemcpy3DParms::srcArray and "
        "hipMemcpy3DParms::dstArray",
        hipErrorInvalidValue);
  CHIPGraphNodeMemcpy *Node = new CHIPGraphNodeMemcpy(pCopyParams);
  Node->addDependencies(DECONST_NODES(pDependencies), numDependencies);
  *pGraphNode = Node;
  GRAPH(graph)->addNode(Node);
  Node->Msg += "Memcpy";
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node,
                                       hipMemcpy3DParms *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  hipMemcpy3DParms Params =
      static_cast<CHIPGraphNodeMemcpy *>(node)->getParams();
  pNodeParams = &Params;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node,
                                       const hipMemcpy3DParms *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  static_cast<CHIPGraphNodeMemcpy *>(node)->setParams(pNodeParams);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec,
                                           hipGraphNode_t node,
                                           hipMemcpy3DParms *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  auto ExecNode =
      EXEC(hGraphExec)->getOriginalGraphPtr()->nodeLookup(NODE(node));
  if (!ExecNode)
    CHIPERR_LOG_AND_THROW("Failed to find the node in hipGraphExec_t",
                          hipErrorInvalidValue);

  auto CastNode = static_cast<CHIPGraphNodeMemcpy *>(node);
  if (!CastNode)
    CHIPERR_LOG_AND_THROW("Node provided failed to cast to CHIPGraphNodeMemcpy",
                          hipErrorInvalidValue);

  CastNode->setParams(const_cast<hipMemcpy3DParms *>(pNodeParams));
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t *pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t *pDependencies,
                                   size_t numDependencies, void *dst,
                                   const void *src, size_t count,
                                   hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphNodeMemcpy *Node = new CHIPGraphNodeMemcpy(dst, src, count, kind);
  *pGraphNode = Node;
  Node->addDependencies(DECONST_NODES(pDependencies), numDependencies);
  GRAPH(graph)->addNode(Node);
  Node->Msg += "Memcpy1D";
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void *dst,
                                         const void *src, size_t count,
                                         hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();
  auto CastNode = static_cast<CHIPGraphNodeMemcpy *>(node);
  if (!CastNode)
    CHIPERR_LOG_AND_THROW("Node provided failed to cast to CHIPGraphNodeMemcpy",
                          hipErrorInvalidValue);

  CastNode->setParams(dst, src, count, kind);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec,
                                             hipGraphNode_t node, void *dst,
                                             const void *src, size_t count,
                                             hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();
  auto ExecNode =
      EXEC(hGraphExec)->getOriginalGraphPtr()->nodeLookup(NODE(node));
  if (!ExecNode)
    CHIPERR_LOG_AND_THROW("Failed to find the node in hipGraphExec_t",
                          hipErrorInvalidValue);

  auto CastNode = static_cast<CHIPGraphNodeMemcpy *>(node);
  if (!CastNode)
    CHIPERR_LOG_AND_THROW("Node provided failed to cast to CHIPGraphNodeMemcpy",
                          hipErrorInvalidValue);

  CastNode->setParams(dst, src, count, kind);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t *pGraphNode,
                                           hipGraph_t graph,
                                           const hipGraphNode_t *pDependencies,
                                           size_t numDependencies, void *dst,
                                           const void *symbol, size_t count,
                                           size_t offset, hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphNodeMemcpyFromSymbol *Node =
      new CHIPGraphNodeMemcpyFromSymbol(dst, symbol, count, offset, kind);
  *pGraphNode = Node;
  Node->addDependencies(DECONST_NODES(pDependencies), numDependencies);
  GRAPH(graph)->addNode(Node);
  Node->Msg += "MemcpyNodeFromSymbol";
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void *dst,
                                                 const void *symbol,
                                                 size_t count, size_t offset,
                                                 hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();
  static_cast<CHIPGraphNodeMemcpyFromSymbol *>(node)->setParams(
      dst, symbol, count, offset, kind);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(
    hipGraphExec_t hGraphExec, hipGraphNode_t node, void *dst,
    const void *symbol, size_t count, size_t offset, hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();
  // Graph obtained from hipGraphExec_t is a clone of the original
  CHIPGraph *Graph = EXEC(hGraphExec)->getOriginalGraphPtr();
  // KernelNode here is a handle to the original
  CHIPGraphNodeMemcpyFromSymbol *KernelNode =
      ((CHIPGraphNodeMemcpyFromSymbol *)node);
  CHIPGraphNodeMemcpyFromSymbol *ExecKernelNode =
      ((CHIPGraphNodeMemcpyFromSymbol *)GRAPH(Graph)->getClonedNodeFromOriginal(
          KernelNode));

  ExecKernelNode->setParams(dst, symbol, count, offset, kind);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t *pGraphNode,
                                         hipGraph_t graph,
                                         const hipGraphNode_t *pDependencies,
                                         size_t numDependencies,
                                         const void *symbol, const void *src,
                                         size_t count, size_t offset,
                                         hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphNodeMemcpyToSymbol *Node = new CHIPGraphNodeMemcpyToSymbol(
      const_cast<void *>(src), symbol, count, offset, kind);
  *pGraphNode = Node;
  Node->addDependencies(DECONST_NODES(pDependencies), numDependencies);
  GRAPH(graph)->addNode(Node);
  Node->Msg += "MemcpyNodeToSymbol";
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node,
                                               const void *symbol,
                                               const void *src, size_t count,
                                               size_t offset,
                                               hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();
  static_cast<CHIPGraphNodeMemcpyToSymbol *>(node)->setParams(
      const_cast<void *>(src), symbol, count, offset, kind);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(
    hipGraphExec_t hGraphExec, hipGraphNode_t node, const void *symbol,
    const void *src, size_t count, size_t offset, hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();
  auto ExecNode =
      EXEC(hGraphExec)->getOriginalGraphPtr()->nodeLookup(NODE(node));
  if (!ExecNode)
    CHIPERR_LOG_AND_THROW("Failed to find the node in hipGraphExec_t",
                          hipErrorInvalidValue);

  auto CastNode = static_cast<CHIPGraphNodeMemcpyToSymbol *>(node);
  if (!CastNode)
    CHIPERR_LOG_AND_THROW(
        "Node provided failed to cast to CHIPGraphNodeMemcpyToSymbol",
        hipErrorInvalidValue);

  CastNode->setParams(const_cast<void *>(src), symbol, count, offset, kind);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphAddMemsetNode(hipGraphNode_t *pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t *pDependencies,
                                 size_t numDependencies,
                                 const hipMemsetParams *pMemsetParams) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphNodeMemset *Node = new CHIPGraphNodeMemset(pMemsetParams);
  Node->addDependencies(DECONST_NODES(pDependencies), numDependencies);
  GRAPH(graph)->addNode(Node);
  *pGraphNode = Node;
  Node->Msg += "Memset";
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node,
                                       hipMemsetParams *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  hipMemsetParams Params =
      static_cast<CHIPGraphNodeMemset *>(node)->getParams();
  *pNodeParams = Params;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node,
                                       const hipMemsetParams *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  static_cast<CHIPGraphNodeMemset *>(node)->setParams(pNodeParams);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec,
                                           hipGraphNode_t node,
                                           const hipMemsetParams *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  auto ExecNode =
      EXEC(hGraphExec)->getOriginalGraphPtr()->nodeLookup(NODE(node));
  if (!ExecNode)
    CHIPERR_LOG_AND_THROW("Failed to find the node in hipGraphExec_t",
                          hipErrorInvalidValue);

  auto CastNode = static_cast<CHIPGraphNodeMemset *>(node);
  if (!CastNode)
    CHIPERR_LOG_AND_THROW("Node provided failed to cast to CHIPGraphNodeMemset",
                          hipErrorInvalidValue);

  CastNode->setParams(pNodeParams);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphAddHostNode(hipGraphNode_t *pGraphNode, hipGraph_t graph,
                               const hipGraphNode_t *pDependencies,
                               size_t numDependencies,
                               const hipHostNodeParams *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphNodeHost *Node = new CHIPGraphNodeHost(pNodeParams);
  Node->addDependencies(DECONST_NODES(pDependencies), numDependencies);
  GRAPH(graph)->addNode(Node);
  *pGraphNode = Node;
  Node->Msg += "Host";
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node,
                                     hipHostNodeParams *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  hipHostNodeParams Params =
      static_cast<CHIPGraphNodeHost *>(node)->getParams();
  *pNodeParams = Params;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node,
                                     const hipHostNodeParams *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  static_cast<CHIPGraphNodeHost *>(node)->setParams(pNodeParams);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec,
                                         hipGraphNode_t node,
                                         const hipHostNodeParams *pNodeParams) {
  CHIP_TRY
  CHIPInitialize();
  auto ExecNode =
      EXEC(hGraphExec)->getOriginalGraphPtr()->nodeLookup(NODE(node));
  if (!ExecNode)
    CHIPERR_LOG_AND_THROW("Failed to find the node in hipGraphExec_t",
                          hipErrorInvalidValue);

  auto CastNode = static_cast<CHIPGraphNodeHost *>(ExecNode);
  if (!CastNode)
    CHIPERR_LOG_AND_THROW("Node provided failed to cast to CHIPGraphNodeMemset",
                          hipErrorInvalidValue);

  CastNode->setParams(pNodeParams);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphAddChildGraphNode(hipGraphNode_t *pGraphNode,
                                     hipGraph_t graph,
                                     const hipGraphNode_t *pDependencies,
                                     size_t numDependencies,
                                     hipGraph_t childGraph) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphNodeGraph *Node = new CHIPGraphNodeGraph(GRAPH(childGraph));
  *pGraphNode = Node;
  Node->addDependencies(DECONST_NODES(pDependencies), numDependencies);
  GRAPH(graph)->addNode(Node);
  Node->Msg += "ChildGraph";
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node,
                                          hipGraph_t *pGraph) {
  CHIP_TRY
  CHIPInitialize();
  *pGraph = static_cast<CHIPGraphNodeGraph *>(node)->getGraph();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec,
                                               hipGraphNode_t node,
                                               hipGraph_t childGraph) {
  CHIP_TRY
  CHIPInitialize();
  static_cast<CHIPGraphNodeGraph *>(node)->setGraph(GRAPH(childGraph));
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphAddEmptyNode(hipGraphNode_t *pGraphNode, hipGraph_t graph,
                                const hipGraphNode_t *pDependencies,
                                size_t numDependencies) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphNodeEmpty *Node = new CHIPGraphNodeEmpty();
  Node->addDependencies(DECONST_NODES(pDependencies), numDependencies);
  *pGraphNode = Node;
  GRAPH(graph)->addNode(Node);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphAddEventRecordNode(hipGraphNode_t *pGraphNode,
                                      hipGraph_t graph,
                                      const hipGraphNode_t *pDependencies,
                                      size_t numDependencies,
                                      hipEvent_t event) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphNodeEventRecord *Node =
      new CHIPGraphNodeEventRecord(static_cast<CHIPEvent *>(event));
  Node->addDependencies(DECONST_NODES(pDependencies), numDependencies);
  *pGraphNode = Node;
  GRAPH(graph)->addNode(Node);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node,
                                           hipEvent_t *event_out) {
  CHIP_TRY
  CHIPInitialize();
  auto CastNode = static_cast<CHIPGraphNodeEventRecord *>(node);
  if (!CastNode)
    CHIPERR_LOG_AND_THROW("Failed to cast CHIPGraphNodeEventRecord",
                          hipErrorInvalidValue);
  *event_out = CastNode->getEvent();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node,
                                           hipEvent_t event) {
  CHIP_TRY
  CHIPInitialize();
  auto CastNode = static_cast<CHIPGraphNodeEventRecord *>(node);
  if (!CastNode)
    CHIPERR_LOG_AND_THROW("Failed to cast CHIPGraphNodeEventRecord",
                          hipErrorInvalidValue);
  CastNode->setEvent(static_cast<CHIPEvent *>(event));
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec,
                                               hipGraphNode_t hNode,
                                               hipEvent_t event) {
  CHIP_TRY
  CHIPInitialize();
  auto ExecNode =
      EXEC(hGraphExec)->getOriginalGraphPtr()->nodeLookup(NODE(hNode));
  if (!ExecNode)
    CHIPERR_LOG_AND_THROW("Failed to find the node in hipGraphExec_t",
                          hipErrorInvalidValue);

  auto CastNode = static_cast<CHIPGraphNodeEventRecord *>(hNode);
  if (!CastNode)
    CHIPERR_LOG_AND_THROW(
        "Node provided failed to cast to CHIPGraphNodeEventRecord",
        hipErrorInvalidValue);

  CastNode->setEvent(static_cast<CHIPEvent *>(event));
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphAddEventWaitNode(hipGraphNode_t *pGraphNode,
                                    hipGraph_t graph,
                                    const hipGraphNode_t *pDependencies,
                                    size_t numDependencies, hipEvent_t event) {
  CHIP_TRY
  CHIPInitialize();
  CHIPGraphNodeWaitEvent *Node =
      new CHIPGraphNodeWaitEvent(static_cast<CHIPEvent *>(event));
  *pGraphNode = Node;
  Node->addDependencies(DECONST_NODES(pDependencies), numDependencies);
  GRAPH(graph)->addNode(Node);
  Node->Msg += "WaitEvent";
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node,
                                         hipEvent_t *event_out) {
  CHIP_TRY
  CHIPInitialize();
  auto CastNode = static_cast<CHIPGraphNodeWaitEvent *>(node);
  if (!CastNode)
    CHIPERR_LOG_AND_THROW(
        "Node provided failed to cast to CHIPGraphNodeWaitEvent",
        hipErrorInvalidValue);

  *event_out = CastNode->getEvent();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node,
                                         hipEvent_t event) {
  CHIP_TRY
  CHIPInitialize();
  auto CastNode = static_cast<CHIPGraphNodeWaitEvent *>(node);
  if (!CastNode)
    CHIPERR_LOG_AND_THROW(
        "Node provided failed to cast to CHIPGraphNodeWaitEvent",
        hipErrorInvalidValue);

  CastNode->setEvent(static_cast<CHIPEvent *>(event));
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec,
                                             hipGraphNode_t hNode,
                                             hipEvent_t event) {
  CHIP_TRY
  CHIPInitialize();
  auto ExecNode =
      EXEC(hGraphExec)->getOriginalGraphPtr()->nodeLookup(NODE(hNode));
  if (!ExecNode)
    CHIPERR_LOG_AND_THROW("Failed to find the node in hipGraphExec_t",
                          hipErrorInvalidValue);

  // TODO Grahs check all of these - somewhere using hNode instead of ExecNode
  auto CastNode = static_cast<CHIPGraphNodeWaitEvent *>(ExecNode);
  if (!CastNode)
    CHIPERR_LOG_AND_THROW(
        "Node provided failed to cast to CHIPGraphNodeWaitEvent",
        hipErrorInvalidValue);

  CastNode->setEvent(static_cast<CHIPEvent *>(event));
  RETURN(hipSuccess);
  CHIP_CATCH
}
hipError_t hipStreamBeginCapture(hipStream_t stream,
                                 hipStreamCaptureMode mode) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = static_cast<CHIPQueue *>(stream);

  if (ChipQueue == Backend->getActiveDevice()->getLegacyDefaultQueue()) {
    RETURN(hipErrorInvalidValue);
  }
  ChipQueue->initCaptureGraph();
  ChipQueue->setCaptureMode(mode);
  ChipQueue->setCaptureStatus(
      hipStreamCaptureStatus::hipStreamCaptureStatusActive);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t *pGraph) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = static_cast<CHIPQueue *>(stream);

  if (ChipQueue == Backend->getActiveDevice()->getLegacyDefaultQueue()) {
    RETURN(hipErrorInvalidValue);
  }
  if (ChipQueue->getCaptureStatus() !=
      hipStreamCaptureStatus::hipStreamCaptureStatusActive) {
    RETURN(hipErrorInvalidValue);
  }
  ChipQueue->setCaptureStatus(
      hipStreamCaptureStatus::hipStreamCaptureStatusNone);
  *pGraph = ChipQueue->getCaptureGraph();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes,
                                   const void *ptr) {
  CHIP_TRY
  CHIPInitialize();

  for (auto Dev : Backend->getDevices()) {
    auto AllocTracker = Dev->AllocationTracker;
    auto AllocInfo = AllocTracker->getAllocInfo(ptr);
    if (AllocInfo) {
      attributes->allocationFlags = AllocInfo->Flags.getRaw();
      attributes->device = AllocInfo->Device;
      attributes->devicePointer = const_cast<void *>(ptr);
      attributes->hostPointer = AllocInfo->HostPtr;
      attributes->isManaged = AllocInfo->Managed;
      attributes->memoryType = AllocInfo->MemoryType;

      // Seems strange but the expected behavior is that if
      // hipPointerGetAttributes gets called with an offset host pointer, the
      // returned attributes should display the offset pointer as the host
      // pointer (as opposed to the base pointer of the allocation)
      attributes->hostPointer = const_cast<void *>(ptr);
      RETURN(hipSuccess);
    }
  }

  RETURN(hipErrorInvalidValue);
  CHIP_CATCH
}

hipError_t hipIpcOpenMemHandle(void **DevPtr, hipIpcMemHandle_t Handle,
                               unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}
hipError_t hipIpcCloseMemHandle(void *DevPtr) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t *Handle, void *DevPtr) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

hipError_t hipMemcpyWithStream(void *Dst, const void *Src, size_t SizeBytes,
                               hipMemcpyKind Kind, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = static_cast<CHIPQueue *>(Stream);
  ChipQueue = Backend->findQueue(ChipQueue);
  if (ChipQueue->getCaptureStatus() != hipStreamCaptureStatusNone) {
    ChipQueue->setCaptureStatus(hipStreamCaptureStatusInvalidated);
    RETURN(hipErrorStreamCaptureInvalidated);
  }

  auto Status = ChipQueue->memCopy(Dst, Src, SizeBytes);
  RETURN(Status);
  CHIP_CATCH
};

hipError_t hipMemcpyPeer(void *Dst, int DstDeviceId, const void *Src,
                         int SrcDeviceId, size_t SizeBytes) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
};
hipError_t hipMemRangeGetAttribute(void *Data, size_t DataSize,
                                   hipMemRangeAttribute Attribute,
                                   const void *DevPtr, size_t Count) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
};

hipError_t hipMemcpyPeerAsync(void *Dst, int DstDeviceId, const void *Src,
                              int SrcDevice, size_t SizeBytes,
                              hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
};

hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D *PCopy,
                                 hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = static_cast<CHIPQueue *>(Stream);

  ChipQueue = Backend->findQueue(ChipQueue);

  if (PCopy->dstPitch == 0)
    return hipSuccess;
  if (PCopy->srcPitch == 0)
    return hipSuccess;
  if (PCopy->Height * PCopy->WidthInBytes == 0)
    return hipSuccess;
  if (PCopy->srcDevice == nullptr && PCopy->dstDevice == nullptr)
    CHIPERR_LOG_AND_THROW("Source and Destination Device pointer is null",
                          hipErrorTbd);

  if (PCopy->dstDevice != nullptr && PCopy->srcDevice == nullptr)
    CHIPERR_LOG_AND_THROW("Source Device pointer is null", hipErrorTbd);
  if (PCopy->srcDevice != nullptr && PCopy->dstDevice == nullptr)
    CHIPERR_LOG_AND_THROW("Source Device pointer is null", hipErrorTbd);

  if ((PCopy->WidthInBytes > PCopy->dstPitch) ||
      (PCopy->WidthInBytes > PCopy->srcPitch))
    CHIPERR_LOG_AND_THROW("Width > src/dest pitches", hipErrorTbd);

  return hipMemcpy2DAsync(PCopy->dstArray->data, PCopy->WidthInBytes,
                          PCopy->srcHost, PCopy->srcPitch, PCopy->WidthInBytes,
                          PCopy->Height, hipMemcpyDefault, ChipQueue);
  CHIP_CATCH
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

hipError_t __hipPushCallConfiguration(dim3 GridDim, dim3 BlockDim,
                                      size_t SharedMem, hipStream_t Stream) {
  logDebug("__hipPushCallConfiguration()");
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = static_cast<CHIPQueue *>(Stream);
  ChipQueue = Backend->findQueue(ChipQueue);

  RETURN(Backend->configureCall(GridDim, BlockDim, SharedMem, ChipQueue));
  CHIP_CATCH
  RETURN(hipSuccess);
}

hipError_t __hipPopCallConfiguration(dim3 *GridDim, dim3 *BlockDim,
                                     size_t *SharedMem, hipStream_t *Stream) {
  logDebug("__hipPopCallConfiguration()");
  CHIP_TRY
  CHIPInitialize();

  auto *ExecItem = ChipExecStack.top();
  ChipExecStack.pop();
  *GridDim = ExecItem->getGrid();
  *BlockDim = ExecItem->getBlock();
  *SharedMem = ExecItem->getSharedMem();
  *Stream = ExecItem->getQueue();
  delete ExecItem;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGetDevice(int *DeviceId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DeviceId);

  CHIPDevice *ChipDev = Backend->getActiveDevice();
  *DeviceId = ChipDev->getDeviceId();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGetDeviceCount(int *Count) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Count);

  *Count = Backend->getNumDevices();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipSetDevice(int DeviceId) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_CHECK_DEVNUM(DeviceId);

  CHIPDevice *SelectedDevice = Backend->getDevices()[DeviceId];
  Backend->setActiveDevice(SelectedDevice);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceSynchronize(void) {
  CHIP_TRY
  CHIPInitialize();

  auto Dev = Backend->getActiveDevice();
  {
    LOCK(Dev->DeviceMtx); // prevents queues from being destryed while iterating
    for (auto Q : Dev->getQueuesNoLock()) {
      Q->finish();
    }
  }

  Backend->getActiveDevice()->getLegacyDefaultQueue()->finish();
  if (Backend->getActiveDevice()->isPerThreadStreamUsed()) {
    Backend->getActiveDevice()->getPerThreadDefaultQueue()->finish();
  }

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceReset(void) {
  CHIP_TRY
  CHIPInitialize();

  CHIPDevice *ChipDev = Backend->getActiveDevice();

  ChipDev->reset();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGet(hipDevice_t *Device, int Ordinal) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Device);
  ERROR_CHECK_DEVNUM(Ordinal);

  /// Since the tests are written such that hipDevice_t is an int, this function
  /// is strange
  *Device = Ordinal;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceComputeCapability(int *Major, int *Minor,
                                      hipDevice_t Device) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Major, Minor);
  ERROR_CHECK_DEVNUM(Device);

  hipDeviceProp_t Props;
  Backend->getDevices()[Device]->copyDeviceProperties(&Props);

  if (Major)
    *Major = Props.major;
  if (Minor)
    *Minor = Props.minor;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetAttribute(int *RetPtr, hipDeviceAttribute_t Attr,
                                 int DeviceId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(RetPtr);
  ERROR_CHECK_DEVNUM(DeviceId);

  *RetPtr = Backend->getDevices()[DeviceId]->getAttr(Attr);
  if (*RetPtr == -1)
    RETURN(hipErrorInvalidValue);
  else
    RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t *Prop, int DeviceId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Prop);
  ERROR_CHECK_DEVNUM(DeviceId);

  Backend->getDevices()[DeviceId]->copyDeviceProperties(Prop);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetLimit(size_t *PValue, enum hipLimit_t Limit) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(PValue);

  auto Device = Backend->getActiveDevice();
  switch (Limit) {
  case hipLimitMallocHeapSize:
    *PValue = Device->getMaxMallocSize();
    break;
  case hipLimitPrintfFifoSize:
    UNIMPLEMENTED(hipErrorNotSupported);
    break;
  default:
    CHIPERR_LOG_AND_THROW("Invalid Limit value", hipErrorInvalidHandle);
  }

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetName(char *Name, int Len, hipDevice_t Device) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Name);
  ERROR_CHECK_DEVNUM(Device);

  std::string DeviceName = (Backend->getDevices()[Device])->getName();

  size_t NameLen = DeviceName.size();
  NameLen = (NameLen < (size_t)Len ? NameLen : Len - 1);
  memcpy(Name, DeviceName.data(), NameLen);
  Name[NameLen] = 0;
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceTotalMem(size_t *Bytes, hipDevice_t Device) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Bytes);
  ERROR_CHECK_DEVNUM(Device);

  if (Bytes)
    *Bytes = (Backend->getDevices()[Device])->getGlobalMemSize();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceSetCacheConfig(hipFuncCache_t CacheCfg) {
  CHIP_TRY
  CHIPInitialize();

  Backend->getActiveDevice()->setCacheConfig(CacheCfg);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetCacheConfig(hipFuncCache_t *CacheCfg) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(CacheCfg);

  if (CacheCfg)
    *CacheCfg = Backend->getActiveDevice()->getCacheConfig();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig *Cfg) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Cfg);

  if (Cfg)
    *Cfg = Backend->getActiveDevice()->getSharedMemConfig();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig Cfg) {
  CHIP_TRY
  CHIPInitialize();

  Backend->getActiveDevice()->setSharedMemConfig(Cfg);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipFuncSetCacheConfig(const void *Func, hipFuncCache_t Cfg) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Func);

  UNIMPLEMENTED(hipErrorNotSupported);
  // RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceGetPCIBusId(char *PciBusId, int Len, int DeviceId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(PciBusId);
  ERROR_CHECK_DEVNUM(DeviceId);
  if (Len < 1)
    RETURN(hipErrorInvalidResourceHandle);

  CHIPDevice *Dev = Backend->getDevices()[DeviceId];

  hipDeviceProp_t Prop;
  Dev->copyDeviceProperties(&Prop);
  snprintf(PciBusId, Len, "%04x:%02x:%02x", Prop.pciDomainID, Prop.pciBusID,
           Prop.pciDeviceID);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceGetByPCIBusId(int *DeviceId, const char *PciBusId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DeviceId, PciBusId);

  int PciDomainID, PciBusID, PciDeviceID;
  int Err =
      sscanf(PciBusId, "%4x:%4x:%4x", &PciDomainID, &PciBusID, &PciDeviceID);
  if (Err == EOF || Err < 3)
    RETURN(hipErrorInvalidValue);
  for (size_t DevIdx = 0; DevIdx < Backend->getNumDevices(); DevIdx++) {
    CHIPDevice *Dev = Backend->getDevices()[DevIdx];
    if (Dev->hasPCIBusId(PciDomainID, PciBusID, PciDeviceID)) {
      *DeviceId = DevIdx;
      RETURN(hipSuccess);
    }
  }

  RETURN(hipErrorInvalidDevice);
  CHIP_CATCH
}

hipError_t hipSetDeviceFlags(unsigned Flags) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceCanAccessPeer(int *CanAccessPeer, int DeviceId,
                                  int PeerDeviceId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(CanAccessPeer);
  ERROR_CHECK_DEVNUM(DeviceId);
  ERROR_CHECK_DEVNUM(PeerDeviceId);

  if (DeviceId == PeerDeviceId) {
    *CanAccessPeer = 0;
    RETURN(hipSuccess);
  }

  CHIPDevice *Dev = Backend->getDevices()[DeviceId];
  CHIPDevice *Peer = Backend->getDevices()[PeerDeviceId];

  *CanAccessPeer = Dev->getPeerAccess(Peer);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceEnablePeerAccess(int PeerDeviceId, unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();

  CHIPDevice *Dev = Backend->getActiveDevice();
  CHIPDevice *Peer = Backend->getDevices()[PeerDeviceId];

  RETURN(Dev->setPeerAccess(Peer, Flags, true));
  CHIP_CATCH
}

hipError_t hipDeviceDisablePeerAccess(int PeerDeviceId) {
  CHIP_TRY
  CHIPInitialize();

  CHIPDevice *Dev = Backend->getActiveDevice();
  CHIPDevice *Peer = Backend->getDevices()[PeerDeviceId];

  RETURN(Dev->setPeerAccess(Peer, 0, false));
  CHIP_CATCH
}

hipError_t hipChooseDevice(int *DeviceId, const hipDeviceProp_t *Prop) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DeviceId, Prop);

  CHIPDevice *Dev = Backend->findDeviceMatchingProps(Prop);
  if (!Dev)
    RETURN(hipErrorInvalidValue);

  *DeviceId = Dev->getDeviceId();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDriverGetVersion(int *DriverVersion) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DriverVersion);

  if (DriverVersion) {
    *DriverVersion = 4;
    logWarn("Driver version is hardcoded to 4");
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);

  CHIP_CATCH
}

hipError_t hipRuntimeGetVersion(int *RuntimeVersion) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(RuntimeVersion);

  if (RuntimeVersion) {
    *RuntimeVersion = 1;
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);

  CHIP_CATCH
}

hipError_t hipGetLastError(void) {
  // No runtime initialization here as the function does not depend on the
  // driver nor the driver affects the answer.
  hipError_t Temp = CHIPTlsLastError;
  CHIPTlsLastError = hipSuccess;
  return Temp;
}

hipError_t hipPeekAtLastError(void) {
  // No runtime initialization here as the function does not depend on the
  // driver nor the driver affects the answer.
  return CHIPTlsLastError;
}

const char *hipGetErrorName(hipError_t HipError) {
  switch (HipError) {
  case hipSuccess:
    return "hipSuccess";
  case hipErrorOutOfMemory:
    return "hipErrorOutOfMemory";
  case hipErrorNotInitialized:
    return "hipErrorNotInitialized";
  case hipErrorDeinitialized:
    return "hipErrorDeinitialized";
  case hipErrorProfilerDisabled:
    return "hipErrorProfilerDisabled";
  case hipErrorProfilerNotInitialized:
    return "hipErrorProfilerNotInitialized";
  case hipErrorProfilerAlreadyStarted:
    return "hipErrorProfilerAlreadyStarted";
  case hipErrorProfilerAlreadyStopped:
    return "hipErrorProfilerAlreadyStopped";
  case hipErrorInvalidImage:
    return "hipErrorInvalidImage";
  case hipErrorInvalidContext:
    return "hipErrorInvalidContext";
  case hipErrorContextAlreadyCurrent:
    return "hipErrorContextAlreadyCurrent";
  case hipErrorMapFailed:
    return "hipErrorMapFailed";
  case hipErrorUnmapFailed:
    return "hipErrorUnmapFailed";
  case hipErrorArrayIsMapped:
    return "hipErrorArrayIsMapped";
  case hipErrorAlreadyMapped:
    return "hipErrorAlreadyMapped";
  case hipErrorNoBinaryForGpu:
    return "hipErrorNoBinaryForGpu";
  case hipErrorAlreadyAcquired:
    return "hipErrorAlreadyAcquired";
  case hipErrorNotMapped:
    return "hipErrorNotMapped";
  case hipErrorNotMappedAsArray:
    return "hipErrorNotMappedAsArray";
  case hipErrorNotMappedAsPointer:
    return "hipErrorNotMappedAsPointer";
  case hipErrorECCNotCorrectable:
    return "hipErrorECCNotCorrectable";
  case hipErrorUnsupportedLimit:
    return "hipErrorUnsupportedLimit";
  case hipErrorContextAlreadyInUse:
    return "hipErrorContextAlreadyInUse";
  case hipErrorPeerAccessUnsupported:
    return "hipErrorPeerAccessUnsupported";
  case hipErrorInvalidKernelFile:
    return "hipErrorInvalidKernelFile";
  case hipErrorInvalidGraphicsContext:
    return "hipErrorInvalidGraphicsContext";
  case hipErrorInvalidSource:
    return "hipErrorInvalidSource";
  case hipErrorFileNotFound:
    return "hipErrorFileNotFound";
  case hipErrorSharedObjectSymbolNotFound:
    return "hipErrorSharedObjectSymbolNotFound";
  case hipErrorSharedObjectInitFailed:
    return "hipErrorSharedObjectInitFailed";
  case hipErrorOperatingSystem:
    return "hipErrorOperatingSystem";
  case hipErrorSetOnActiveProcess:
    return "hipErrorSetOnActiveProcess";
  case hipErrorInvalidHandle:
    return "hipErrorInvalidHandle";
  case hipErrorNotFound:
    return "hipErrorNotFound";
  case hipErrorIllegalAddress:
    return "hipErrorIllegalAddress";
  case hipErrorInvalidSymbol:
    return "hipErrorInvalidSymbol";
  case hipErrorMissingConfiguration:
    return "hipErrorMissingConfiguration";
  case hipErrorLaunchFailure:
    return "hipErrorLaunchFailure";
  case hipErrorPriorLaunchFailure:
    return "hipErrorPriorLaunchFailure";
  case hipErrorLaunchTimeOut:
    return "hipErrorLaunchTimeOut";
  case hipErrorLaunchOutOfResources:
    return "hipErrorLaunchOutOfResources";
  case hipErrorInvalidDeviceFunction:
    return "hipErrorInvalidDeviceFunction";
  case hipErrorInvalidConfiguration:
    return "hipErrorInvalidConfiguration";
  case hipErrorInvalidDevice:
    return "hipErrorInvalidDevice";
  case hipErrorInvalidValue:
    return "hipErrorInvalidValue";
  case hipErrorInvalidDevicePointer:
    return "hipErrorInvalidDevicePointer";
  case hipErrorInvalidMemcpyDirection:
    return "hipErrorInvalidMemcpyDirection";
  case hipErrorUnknown:
    return "hipErrorUnknown";
  case hipErrorNotReady:
    return "hipErrorNotReady";
  case hipErrorNoDevice:
    return "hipErrorNoDevice";
  case hipErrorPeerAccessAlreadyEnabled:
    return "hipErrorPeerAccessAlreadyEnabled";
  case hipErrorNotSupported:
    return "hipErrorNotSupported";
  case hipErrorPeerAccessNotEnabled:
    return "hipErrorPeerAccessNotEnabled";
  case hipErrorRuntimeMemory:
    return "hipErrorRuntimeMemory";
  case hipErrorRuntimeOther:
    return "hipErrorRuntimeOther";
  case hipErrorHostMemoryAlreadyRegistered:
    return "hipErrorHostMemoryAlreadyRegistered";
  case hipErrorHostMemoryNotRegistered:
    return "hipErrorHostMemoryNotRegistered";
  case hipErrorTbd:
    return "hipErrorTbd";
  default:
    return "hipErrorUnknown";
  }
}

const char *hipGetErrorString(hipError_t HipError) {
  return hipGetErrorName(HipError);
}

hipError_t hipStreamCreate(hipStream_t *Stream) {
  RETURN(hipStreamCreateWithFlags(Stream, 0));
}

hipError_t hipStreamCreateWithFlags(hipStream_t *Stream, unsigned int Flags) {
  RETURN(hipStreamCreateWithPriority(Stream, Flags, 1));
}

hipError_t hipStreamCreateWithPriority(hipStream_t *Stream, unsigned int Flags,
                                       int Priority) {
  CHIP_TRY
  CHIPInitialize();
  if (Stream == nullptr)
    CHIPERR_LOG_AND_THROW("Stream pointer is null", hipErrorInvalidValue);

  CHIPDevice *Dev = Backend->getActiveDevice();

  CHIPQueueFlags FlagsParsed{Flags};

  // Clamp priority between min and max
  auto MaxPriority = 0;
  auto MinPriority = Backend->getQueuePriorityRange();
  auto ClampedPriority = std::min(MinPriority, std::max(MaxPriority, Priority));
  CHIPQueue *ChipQueue =
      Dev->createQueueAndRegister(FlagsParsed, ClampedPriority);
  *Stream = ChipQueue;
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceGetStreamPriorityRange(int *LeastPriority,
                                           int *GreatestPriority) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(LeastPriority, GreatestPriority);

  if (LeastPriority)
    *LeastPriority = Backend->getQueuePriorityRange();
  if (GreatestPriority)
    *GreatestPriority = 0;
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipStreamDestroy(hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = static_cast<CHIPQueue *>(Stream);
  if (ChipQueue == hipStreamPerThread)
    CHIPERR_LOG_AND_THROW("Attemped to destroy default per-thread queue",
                          hipErrorTbd);

  if (ChipQueue == hipStreamLegacy)
    CHIPERR_LOG_AND_THROW("Attemped to destroy default legacy queue",
                          hipErrorTbd);

  ChipQueue = Backend->findQueue(ChipQueue);
  if (ChipQueue->getCaptureStatus() != hipStreamCaptureStatusNone) {
    ChipQueue->setCaptureStatus(hipStreamCaptureStatusInvalidated);
    RETURN(hipErrorStreamCaptureInvalidated);
  }

  CHIPDevice *Dev = Backend->getActiveDevice();

  // make sure nothing is pending in the stream
  ChipQueue->finish();

  if (Dev->removeQueue(ChipQueue))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);

  CHIP_CATCH
}

hipError_t hipStreamQuery(hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = static_cast<CHIPQueue *>(Stream);
  ChipQueue = Backend->findQueue(ChipQueue);
  if (ChipQueue->getCaptureStatus() != hipStreamCaptureStatusNone) {
    ChipQueue->setCaptureStatus(hipStreamCaptureStatusInvalidated);
    RETURN(hipErrorStreamCaptureInvalidated);
  }

  if (ChipQueue->query()) {
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorNotReady);

  CHIP_CATCH
}

hipError_t hipStreamSynchronize(hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = static_cast<CHIPQueue *>(Stream);
  ChipQueue = Backend->findQueue(ChipQueue);
  if (ChipQueue->getCaptureStatus() != hipStreamCaptureStatusNone) {
    ChipQueue->setCaptureStatus(hipStreamCaptureStatusInvalidated);
    RETURN(hipErrorStreamCaptureInvalidated);
  }

  Backend->getActiveDevice()->getContext()->syncQueues(ChipQueue);
  ChipQueue->finish();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipStreamWaitEvent(hipStream_t Stream, hipEvent_t Event,
                              unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = static_cast<CHIPQueue *>(Stream);
  auto ChipEvent = static_cast<CHIPEvent *>(Event);

  ChipQueue = Backend->findQueue(ChipQueue);
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeWaitEvent>(ChipEvent)) {
    RETURN(hipSuccess);
  }
  ERROR_IF((ChipQueue == nullptr), hipErrorInvalidResourceHandle);
  ERROR_IF((Event == nullptr), hipErrorInvalidResourceHandle);

  // Unless the event is in recording state, we can't wait on it
  if (ChipEvent->getEventStatus() == EVENT_STATUS_INIT) {
    RETURN(hipSuccess);
  }
  std::shared_ptr<CHIPEvent> ChipEventShared =
      Backend->userEventLookup(ChipEvent);
  std::vector<std::shared_ptr<CHIPEvent>> EventsToWaitOn;
  if (ChipEventShared.get())
    EventsToWaitOn.push_back(ChipEventShared);
  auto BarrierEvent = ChipQueue->enqueueBarrier(EventsToWaitOn);
  BarrierEvent->Msg = "hipStreamWaitEvent-enqueueBarrier";
  RETURN(hipSuccess);
  CHIP_CATCH
}

int hipGetStreamDeviceId(hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  CHIPDevice *Device =
      Backend->findQueue(static_cast<CHIPQueue *>(Stream))->getDevice();
  return Device->getDeviceId();
  CHIP_CATCH
}

hipError_t hipStreamGetFlags(hipStream_t Stream, unsigned int *Flags) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Flags);
  auto ChipQueue = static_cast<CHIPQueue *>(Stream);
  ChipQueue = Backend->findQueue(ChipQueue);
  if (ChipQueue->getCaptureStatus() != hipStreamCaptureStatusNone) {
    ChipQueue->setCaptureStatus(hipStreamCaptureStatusInvalidated);
    RETURN(hipErrorStreamCaptureInvalidated);
  }

  auto ChipQueueFlags = ChipQueue->getFlags();
  *Flags = ChipQueueFlags.getRaw();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipStreamGetPriority(hipStream_t Stream, int *Priority) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = static_cast<CHIPQueue *>(Stream);
  if (Priority == nullptr) {
    CHIPERR_LOG_AND_THROW("Priority is nullptr", hipErrorInvalidValue);
  }

  ChipQueue = Backend->findQueue(ChipQueue);
  if (ChipQueue->getCaptureStatus() != hipStreamCaptureStatusNone) {
    ChipQueue->setCaptureStatus(hipStreamCaptureStatusInvalidated);
    RETURN(hipErrorStreamCaptureInvalidated);
  }

  *Priority = ChipQueue->getPriority();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipStreamAddCallback(hipStream_t Stream,
                                hipStreamCallback_t Callback, void *UserData,
                                unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  if (Flags)
    CHIPERR_LOG_AND_THROW(
        "hipStreamAddCallback: Flags are non-zero (reserved argument. Must be "
        "0)",
        hipErrorTbd);
  // TODO: Can't use NULLCHECK for this one
  if (Callback == nullptr)
    CHIPERR_LOG_AND_THROW("passed in nullptr", hipErrorInvalidValue);
  auto ChipQueue = static_cast<CHIPQueue *>(Stream);
  ChipQueue = Backend->findQueue(ChipQueue);
  if (ChipQueue->getCaptureStatus() != hipStreamCaptureStatusNone) {
    ChipQueue->setCaptureStatus(hipStreamCaptureStatusInvalidated);
    RETURN(hipErrorStreamCaptureInvalidated);
  }

  ChipQueue->addCallback(Callback, UserData);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t *Base, size_t *Size,
                                 hipDeviceptr_t Ptr) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Base, Size, Ptr);

  auto AllocTracker = Backend->getActiveDevice()->AllocationTracker;
  auto AllocInfo = AllocTracker->getAllocInfo(Ptr);
  if (!AllocInfo)
    RETURN(hipErrorInvalidValue);

  *Base = AllocInfo->DevPtr;
  *Size = AllocInfo->Size;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipEventCreate(hipEvent_t *Event) {
  RETURN(hipEventCreateWithFlags(Event, 0));
}

hipError_t hipEventCreateWithFlags(hipEvent_t *Event, unsigned Flags) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Event);

  CHIPEventFlags EventFlags{Flags};

  auto ChipEvent =
      Backend->createCHIPEvent(Backend->getActiveContext(), EventFlags, true);
  {
    LOCK(Backend->UserEventsMtx);
    Backend->UserEvents.push_back(ChipEvent);
  }

  *Event = ChipEvent.get();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipEventRecord(hipEvent_t Event, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  // TODO: Why does this check fail for OpenCL but not for Level0
  NULLCHECK(Event);
  auto ChipEvent = static_cast<CHIPEvent *>(Event);
  auto ChipQueue = static_cast<CHIPQueue *>(Stream);
  ChipQueue = Backend->findQueue(ChipQueue);
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeEventRecord>(ChipEvent)) {
    RETURN(hipSuccess);
  }

  ChipEvent->recordStream(ChipQueue);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipEventDestroy(hipEvent_t Event) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Event);
  CHIPEvent *ChipEvent = static_cast<CHIPEvent *>(Event);

  LOCK(Backend->UserEventsMtx);
  Backend->UserEvents.erase(
      std::remove_if(Backend->UserEvents.begin(), Backend->UserEvents.end(),
                     [&ChipEvent](const std::shared_ptr<CHIPEvent> &x) {
                       return x.get() == ChipEvent;
                     }),
      Backend->UserEvents.end());

  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipEventSynchronize(hipEvent_t Event) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Event);
  CHIPEvent *ChipEvent = static_cast<CHIPEvent *>(Event);

  ChipEvent->wait();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipEventElapsedTime(float *Ms, hipEvent_t Start, hipEvent_t Stop) {
  CHIP_TRY
  CHIPInitialize();
  if (!Ms)
    CHIPERR_LOG_AND_THROW("Ms pointer is null", hipErrorInvalidValue);
  NULLCHECK(Start, Stop);
  CHIPEvent *ChipEventStart = static_cast<CHIPEvent *>(Start);
  CHIPEvent *ChipEventStop = static_cast<CHIPEvent *>(Stop);
  if (!ChipEventStart->isRecordingOrRecorded() ||
      !ChipEventStop->isRecordingOrRecorded()) {
    CHIPERR_LOG_AND_THROW("One of the events was not recorded",
                          hipErrorInvalidHandle);
  }
  if (ChipEventStart->getFlags().isDisableTiming() ||
      ChipEventStop->getFlags().isDisableTiming())
    CHIPERR_LOG_AND_THROW("One of the events has timings disabled. "
                          "Unable to return elasped time",
                          hipErrorInvalidResourceHandle);

  *Ms = ChipEventStart->getElapsedTime(ChipEventStop);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipEventQuery(hipEvent_t Event) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Event);
  CHIPEvent *ChipEvent = static_cast<CHIPEvent *>(Event);

  ChipEvent->updateFinishStatus();
  if (ChipEvent->isFinished())
    RETURN(hipSuccess);

  RETURN(hipErrorNotReady);

  CHIP_CATCH
}

hipError_t hipMalloc(void **Ptr, size_t Size) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Ptr);

  if (Size == 0) {
    *Ptr = nullptr;
    RETURN(hipSuccess);
  }
  void *RetVal = Backend->getActiveContext()->allocate(
      Size, hipMemoryType::hipMemoryTypeDevice);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  *Ptr = RetVal;
  logInfo("hipMalloc(ptr={}, size={})", (void *)RetVal, Size);

  // currently required for PVC
  bool firstTouch;
  auto Status = Backend->getActiveDevice()->getDefaultQueue()->memCopy(
      RetVal, &firstTouch, 1);
  assert(Status == hipSuccess);

  RETURN(hipSuccess);

  CHIP_CATCH
}
hipError_t hipMallocManaged(void **DevPtr, size_t Size, unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DevPtr);

  // TODO: Create a class for parsing this, default to attach global
  // attach host should be device allocate with associated host poitner?
  auto FlagsParsed = CHIPManagedMemFlags{Flags};
  switch (FlagsParsed) {
  case CHIPManagedMemFlags::AttachGlobal:
    break;
  case CHIPManagedMemFlags::AttachHost:
    break;
  default:
    CHIPERR_LOG_AND_THROW("Invalid value passed for hipMallocManaged flags",
                          hipErrorInvalidValue);
  }

  if (Size < 0)
    CHIPERR_LOG_AND_THROW("Negative Allocation size",
                          hipErrorInvalidResourceHandle);

  if (Size == 0) {
    *DevPtr = nullptr;
    RETURN(hipSuccess);
  }

  void *RetVal = Backend->getActiveDevice()->getContext()->allocate(
      Size, hipMemoryType::hipMemoryTypeUnified);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  *DevPtr = RetVal;
  RETURN(hipSuccess);

  CHIP_CATCH
};

DEPRECATED("use hipHostMalloc instead")
hipError_t hipMallocHost(void **Ptr, size_t Size) {
  RETURN(hipMalloc(Ptr, Size));
}

hipError_t hipHostMalloc(void **Ptr, size_t Size, unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  if (Ptr == nullptr)
    CHIPERR_LOG_AND_THROW("Ptr is null", hipErrorInvalidValue);
  if (Size == 0) {
    *Ptr = nullptr;
    RETURN(hipSuccess);
  }

  auto FlagsParsed = CHIPHostAllocFlags(Flags);

  void *RetVal = Backend->getActiveContext()->allocate(
      Size, 0x1000, hipMemoryType::hipMemoryTypeHost, FlagsParsed);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  int PageLockSuccess = mlock(RetVal, Size);
  if (PageLockSuccess != 0)
    logCritical("Page Lock failure {}", errno);
  assert(PageLockSuccess == 0 && "Failed to page lock memory");

  *Ptr = RetVal;
  RETURN(hipSuccess);
  CHIP_CATCH
}

DEPRECATED("use hipHostMalloc instead")
hipError_t hipHostAlloc(void **Ptr, size_t Size, unsigned int Flags) {
  RETURN(hipHostMalloc(Ptr, Size, Flags));
}

hipError_t hipFree(void *Ptr) {
  CHIP_TRY
  CHIPInitialize();
  logInfo("hipFree(ptr={})", (void *)Ptr);

  auto Status = hipDeviceSynchronize();
  ERROR_IF((Status != hipSuccess), hipErrorTbd);

  if (Ptr == nullptr)
    RETURN(hipSuccess);
  RETURN(Backend->getActiveContext()->free(Ptr));

  CHIP_CATCH
}

hipError_t hipHostFree(void *Ptr) {
  int Status = munlock(Ptr, 0);
  assert(Status == 0 && "Failed to unlock page-locked memory");

  auto *AllocTracker = Backend->getActiveDevice()->AllocationTracker;
  auto *AllocInfo = AllocTracker->getAllocInfo(Ptr);
  if (AllocInfo && AllocInfo->IsHostRegistered)
    RETURN(hipErrorInvalidValue); // Must use hipHostUnregister() instead.

  RETURN(hipFree(Ptr));
}

DEPRECATED("use hipHostFree instead")
hipError_t hipFreeHost(void *Ptr) { RETURN(hipHostFree(Ptr)); }

hipError_t hipMemPrefetchAsync(const void *Ptr, size_t Count, int DstDevId,
                               hipStream_t Stream) {
  CHIP_TRY
  UNIMPLEMENTED(hipErrorTbd);
  CHIPInitialize();
  NULLCHECK(Ptr);
  auto ChipQueue = static_cast<CHIPQueue *>(Stream);
  ChipQueue = Backend->findQueue(ChipQueue);
  // TODO Graphs - async operation should be supported by graphs but no prefetch
  // node is defined
  ERROR_CHECK_DEVNUM(DstDevId);
  CHIPDevice *Dev = Backend->getDevices()[DstDevId];

  // Check if given Stream belongs to the requested device
  ERROR_IF(ChipQueue->getDevice() != Dev, hipErrorInvalidDevice);
  ChipQueue->memPrefetch(Ptr, Count);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemAdvise(const void *Ptr, size_t Count, hipMemoryAdvise Advice,
                        int DstDevId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Ptr);

  if (Ptr == 0 || Count == 0) {
    RETURN(hipSuccess);
  }

  UNIMPLEMENTED(hipErrorNotSupported);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipHostGetDevicePointer(void **DevPtr, void *HostPtr,
                                   unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DevPtr, HostPtr);

  auto Device = Backend->getActiveDevice();
  auto AllocInfo = Device->AllocationTracker->getAllocInfo(HostPtr);
  if (!AllocInfo)
    CHIPERR_LOG_AND_THROW("Host pointer is not allocated by hipHostMalloc or "
                          "registered with hipHostRegister!",
                          hipErrorInvalidValue);

  *DevPtr = AllocInfo->DevPtr;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipHostGetFlags(unsigned int *FlagsPtr, void *HostPtr) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(FlagsPtr, HostPtr);

  auto AllocTracker = Backend->getActiveDevice()->AllocationTracker;
  auto AllocInfo = AllocTracker->getAllocInfo(HostPtr);

  *FlagsPtr = AllocInfo->Flags.getRaw();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipHostRegister(void *HostPtr, size_t SizeBytes,
                           unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  if (!HostPtr || !SizeBytes)
    RETURN(hipErrorInvalidValue);

  // TODO fixOpenCLTests - make this a class
  if (Flags) {
    // Currently, the flags are ignored. This only exists to satisfy hip-tests.

    // First 4 bits are valid flag bits. This includes flags from CUDA which are
    // not supported or documented in HIP.
    constexpr unsigned FlagMask = (1u << 4u) - 1u;

    if (Flags & ~FlagMask) // Has invalid flags
      CHIPERR_LOG_AND_THROW("Invalid hipHostRegister flags passed",
                             hipErrorInvalidValue);

    if (Flags & hipHostRegisterIoMemory)
      CHIPERR_LOG_AND_THROW("Unsupported hipHostRegisterIoMemory flag",
                             hipErrorInvalidValue);
  }

  void *DevPtr;
  if (hipMalloc(&DevPtr, SizeBytes) != hipSuccess)
    // Translate hipOutOfMemory to hipErrorInvalidValue. The latter is
    // the one hip-tests suite expects in case of OoM.
    RETURN(hipErrorInvalidValue);

  // Associate the pointer
  auto Device = Backend->getActiveDevice();
  // TODO fixOpenCLTests - use recordAllocation()
  Device->AllocationTracker->registerHostPointer(HostPtr, DevPtr);

  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipHostUnregister(void *HostPtr) {
  CHIP_TRY
  CHIPInitialize();
  if (!HostPtr)
    CHIPERR_LOG_AND_THROW("Host pointer is nullptr!", hipErrorInvalidValue);

  auto *Device = Backend->getActiveDevice();
  auto *AllocInfo = Device->AllocationTracker->getAllocInfo(HostPtr);
  if (!AllocInfo)
    CHIPERR_LOG_AND_THROW("Host pointer is not registered!",
                          hipErrorHostMemoryNotRegistered);
  auto Err = hipFree(AllocInfo->DevPtr);
  RETURN(Err);

  CHIP_CATCH
}

static hipError_t hipMallocPitch3D(void **Ptr, size_t *Pitch, size_t Width,
                                   size_t Height, size_t Depth) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Ptr, Pitch);

  *Pitch = ((((int)Width - 1) / SVM_ALIGNMENT) + 1) * SVM_ALIGNMENT;
  const size_t SizeBytes =
      (*Pitch) * std::max<size_t>(1, Height) * std::max<size_t>(1, Depth);

  void *RetVal = Backend->getActiveContext()->allocate(
      SizeBytes, hipMemoryType::hipMemoryTypeDevice);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  *Ptr = RetVal;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMallocPitch(void **Ptr, size_t *Pitch, size_t Width,
                          size_t Height) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Ptr, Pitch);

  RETURN(hipMallocPitch3D(Ptr, Pitch, Width, Height, 0));

  CHIP_CATCH
}

hipError_t hipMalloc3DArray(hipArray **Array,
                            const struct hipChannelFormatDesc *Desc,
                            struct hipExtent Extent, unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Array, Desc);

  auto Width = Extent.width;
  auto Height = Extent.height;
  auto Depth = Extent.depth;

  ERROR_IF((Width == 0), hipErrorInvalidValue);

  *Array = new hipArray;
  ERROR_IF((*Array == nullptr), hipErrorOutOfMemory);

  auto TexType = hipTextureType1D;
  if (Depth) {
    TexType = hipTextureType3D;
  } else if (Height) {
    TexType = hipTextureType2D;
  }
  hipArray_Format hipArrayFormatArray;
  switch (Desc->f) {
  case hipChannelFormatKindSigned:
    hipArrayFormatArray = HIP_AD_FORMAT_SIGNED_INT32;
    break;

  case hipChannelFormatKindUnsigned:
    hipArrayFormatArray = HIP_AD_FORMAT_UNSIGNED_INT32;
    break;

  case hipChannelFormatKindFloat:
    hipArrayFormatArray = HIP_AD_FORMAT_FLOAT;
    break;

  case hipChannelFormatKindNone:
    CHIPERR_LOG_AND_THROW("hipChannelFormatKindNone?", hipErrorInvalidValue);
    break;

  default:
    CHIPERR_LOG_AND_THROW("Invalid channel format", hipErrorInvalidValue);
  }

  (*Array)->data = nullptr;
  (*Array)->desc = *Desc;
  (*Array)->type = hipArrayDefault;
  (*Array)->width = Width;
  (*Array)->height = Height;
  (*Array)->depth = Depth;
  (*Array)->Format = hipArrayFormatArray;
  (*Array)->NumChannels = 1;
  (*Array)->isDrv = false;
  (*Array)->textureType = TexType;
  void **Ptr = &Array[0]->data;

  size_t AllocSize = Width * std::max<size_t>(Height, 1) *
                     std::max<size_t>(Depth, 1) * getChannelByteSize(*Desc);

  void *RetVal = Backend->getActiveContext()->allocate(
      AllocSize, hipMemoryType::hipMemoryTypeDevice);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  *Ptr = RetVal;
  RETURN(hipSuccess);
  CHIP_CATCH
};

hipError_t hipMallocArray(hipArray **Array, const hipChannelFormatDesc *Desc,
                          size_t Width, size_t Height, unsigned int Flags) {

  // TODO: Sink the logic here into hipMalloc3DArray and call it when
  // it is implemented.
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Array, Desc);

  ERROR_IF((Width == 0), hipErrorInvalidValue);

  *Array = new hipArray;
  ERROR_IF((*Array == nullptr), hipErrorOutOfMemory);

  auto TexType = Height ? hipTextureType2D : hipTextureType1D;
  Array[0]->type = Flags;
  Array[0]->width = Width;
  Array[0]->height = Height;
  Array[0]->depth = 0;
  Array[0]->desc = *Desc;
  Array[0]->isDrv = false;
  Array[0]->textureType = TexType;
  void **Ptr = &Array[0]->data;

  size_t AllocSize =
      Width * std::max<size_t>(Height, 1) * getChannelByteSize(*Desc);

  void *RetVal = Backend->getActiveContext()->allocate(
      AllocSize, hipMemoryType::hipMemoryTypeDevice);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  *Ptr = RetVal;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipArrayCreate(hipArray **Array,
                          const HIP_ARRAY_DESCRIPTOR *AllocateArray) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Array, AllocateArray);

  ERROR_IF((AllocateArray->Width == 0), hipErrorInvalidValue);

  *Array = new hipArray;
  ERROR_IF((*Array == nullptr), hipErrorOutOfMemory);

  Array[0]->width = AllocateArray->Width;
  Array[0]->height = AllocateArray->Height;
  Array[0]->isDrv = true;
  Array[0]->textureType = hipTextureType2D;
  void **Ptr = &Array[0]->data;

  size_t Size = AllocateArray->Width;
  if (AllocateArray->Height > 0) {
    Size = Size * AllocateArray->Height;
  }
  size_t AllocSize = 0;
  switch (AllocateArray->Format) {
  case HIP_AD_FORMAT_UNSIGNED_INT8:
    AllocSize = Size * sizeof(uint8_t);
    break;
  case HIP_AD_FORMAT_UNSIGNED_INT16:
    AllocSize = Size * sizeof(uint16_t);
    break;
  case HIP_AD_FORMAT_UNSIGNED_INT32:
    AllocSize = Size * sizeof(uint32_t);
    break;
  case HIP_AD_FORMAT_SIGNED_INT8:
    AllocSize = Size * sizeof(int8_t);
    break;
  case HIP_AD_FORMAT_SIGNED_INT16:
    AllocSize = Size * sizeof(int16_t);
    break;
  case HIP_AD_FORMAT_SIGNED_INT32:
    AllocSize = Size * sizeof(int32_t);
    break;
  case HIP_AD_FORMAT_HALF:
    AllocSize = Size * sizeof(int16_t);
    break;
  case HIP_AD_FORMAT_FLOAT:
    AllocSize = Size * sizeof(float);
    break;
  default:
    AllocSize = Size;
    break;
  }

  void *RetVal = Backend->getActiveContext()->allocate(
      AllocSize, hipMemoryType::hipMemoryTypeDevice);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  *Ptr = RetVal;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipFreeArray(hipArray *Array) {
  CHIP_TRY
  CHIPInitialize();
  if (!Array || !Array->data)
    RETURN(hipErrorInvalidValue);

  hipError_t Err = hipFree(Array->data);
  if (Err != hipSuccess)
    // HIP test suite expects this but HIP API doc doesn't even list it as
    // one of the possible error codes.
    RETURN(hipErrorContextIsDestroyed);
  delete Array;
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipMalloc3D(hipPitchedPtr *PitchedDevPtr, hipExtent Extent) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(PitchedDevPtr);

  ERROR_IF((Extent.width == 0 || Extent.height == 0), hipErrorInvalidValue);
  ERROR_IF((PitchedDevPtr == nullptr), hipErrorInvalidValue);

  size_t Pitch;

  hipError_t HipStatus = hipMallocPitch3D(
      &PitchedDevPtr->ptr, &Pitch, Extent.width, Extent.height, Extent.depth);

  if (HipStatus == hipSuccess) {
    PitchedDevPtr->pitch = Pitch;
    PitchedDevPtr->xsize = Extent.width;
    PitchedDevPtr->ysize = Extent.height;
  }
  RETURN(HipStatus);

  CHIP_CATCH
}

hipError_t hipMemGetInfo(size_t *Free, size_t *Total) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Free, Total);

  ERROR_IF((Total == nullptr || Free == nullptr), hipErrorInvalidValue);

  auto Dev = Backend->getActiveDevice();
  *Total = Dev->getGlobalMemSize();
  assert(Dev->getGlobalMemSize() > Dev->getUsedGlobalMem());
  *Free = Dev->getGlobalMemSize() - Dev->getUsedGlobalMem();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemPtrGetInfo(void *Ptr, size_t *Size) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Ptr, Size);

  AllocationInfo *AllocInfo =
      Backend->getActiveDevice()->AllocationTracker->getAllocInfo(Ptr);
  *Size = AllocInfo->Size;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyAsync(void *Dst, const void *Src, size_t SizeBytes,
                          hipMemcpyKind Kind, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();

  if (SizeBytes == 0)
    RETURN(hipSuccess);
  NULLCHECK(Dst, Src);

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemcpy>(Dst, Src, SizeBytes,
                                                       Kind)) {
    RETURN(hipSuccess);
  }

  if (Kind == hipMemcpyHostToHost) {
    memcpy(Dst, Src, SizeBytes);
    RETURN(hipSuccess);
  } else {
    ChipQueue->memCopyAsync(Dst, Src, SizeBytes);
    RETURN(hipSuccess);
  }

  CHIP_CATCH
}

hipError_t hipMemcpy(void *Dst, const void *Src, size_t SizeBytes,
                     hipMemcpyKind Kind) {
  CHIP_TRY
  CHIPInitialize();
  logInfo("hipMemcpy Dst={} Src={} Size={} Kind={}", Dst, Src, SizeBytes,
          hipMemcpyKindToString(Kind));

  if (SizeBytes == 0)
    RETURN(hipSuccess);

  NULLCHECK(Dst, Src);

  if (Dst == Src) {
    logWarn("Src and Dst are same. Skipping the copy");
    RETURN(hipSuccess);
  }

  if (Kind == hipMemcpyHostToHost) {
    memcpy(Dst, Src, SizeBytes);
    RETURN(hipSuccess);
  }

  RETURN(Backend->getActiveDevice()->getDefaultQueue()->memCopy(Dst, Src,
                                                                SizeBytes));

  CHIP_CATCH
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t Dst, hipDeviceptr_t Src,
                              size_t SizeBytes, hipStream_t Stream) {
  RETURN(hipMemcpyAsync(Dst, Src, SizeBytes, hipMemcpyDeviceToDevice, Stream));
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t Dst, hipDeviceptr_t Src,
                         size_t SizeBytes) {
  RETURN(hipMemcpy(Dst, Src, SizeBytes, hipMemcpyDeviceToDevice));
}

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t Dst, void *Src, size_t SizeBytes,
                              hipStream_t Stream) {
  RETURN(hipMemcpyAsync(Dst, Src, SizeBytes, hipMemcpyHostToDevice, Stream));
}

hipError_t hipMemcpyHtoD(hipDeviceptr_t Dst, void *Src, size_t SizeBytes) {
  RETURN(hipMemcpy(Dst, Src, SizeBytes, hipMemcpyHostToDevice));
}

hipError_t hipMemcpyDtoHAsync(void *Dst, hipDeviceptr_t Src, size_t SizeBytes,
                              hipStream_t Stream) {
  RETURN(hipMemcpyAsync(Dst, Src, SizeBytes, hipMemcpyDeviceToHost, Stream));
}

hipError_t hipMemcpyDtoH(void *Dst, hipDeviceptr_t Src, size_t SizeBytes) {
  RETURN(hipMemcpy(Dst, Src, SizeBytes, hipMemcpyDeviceToHost));
}

hipError_t hipMemset2DAsync(void *Dst, size_t Pitch, int Value, size_t Width,
                            size_t Height, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst);

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  const hipMemsetParams Params = {
      /* Dst */ Dst,
      /* elementSize*/ sizeof(int),
      /* height */ Height,
      /* pitch */ Pitch,
      /* value */ (unsigned int)Value, /* TODO Graphs - why is the arg for
                                          memset unsigned? */
      /* width */ Width};
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemset>(Params)) {
    RETURN(hipSuccess);
  }

  hipError_t Res = hipSuccess;
  LOCK(ChipQueue->QueueMtx); // prevent interruptions
  for (size_t i = 0; i < Height; i++) {
    size_t SizeBytes = Width * sizeof(int);
    auto Offset = Pitch * i;
    char *DstP = (char *)Dst;
    auto Res = hipMemsetAsync(DstP + Offset, Value, SizeBytes, Stream);
    if (Res != hipSuccess)
      break;
  }

  RETURN(Res);
  CHIP_CATCH
}

hipError_t hipMemset2D(void *Dst, size_t Pitch, int Value, size_t Width,
                       size_t Height) {
  CHIP_TRY
  CHIPInitialize();

  auto ChipQueue = Backend->getActiveDevice()->getDefaultQueue();
  auto Res = hipMemset2DAsync(Dst, Pitch, Value, Width, Height, ChipQueue);
  if (Res == hipSuccess)
    ChipQueue->finish();

  RETURN(Res);
  CHIP_CATCH
}

hipError_t hipMemset3DAsync(hipPitchedPtr PitchedDevPtr, int Value,
                            hipExtent Extent, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(PitchedDevPtr.ptr);

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  const hipMemsetParams Params = {
      /* Dst */ PitchedDevPtr.ptr,
      /* elementSize*/ sizeof(int),
      /* height */ Extent.height,
      /* pitch */ PitchedDevPtr.pitch,
      /* value */ (unsigned int)Value, /* TODO Graphs - why is the arg for
                                          memset unsigned? */
      /* width */ Extent.width};
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemset>(Params)) {
    RETURN(hipSuccess);
  }

  if (Extent.height * Extent.width * Extent.depth == 0)
    return hipSuccess;

  if (Extent.height > PitchedDevPtr.ysize ||
      Extent.width > PitchedDevPtr.xsize || Extent.depth > PitchedDevPtr.pitch)
    CHIPERR_LOG_AND_THROW("Extent exceeds allocation", hipErrorTbd);

  // Check if pointer inside allocation range
  auto AllocTracker = ChipQueue->getDevice()->AllocationTracker;
  AllocationInfo *AllocInfo =
      AllocTracker->getAllocInfoCheckPtrRanges(PitchedDevPtr.ptr);
  if (!AllocInfo)
    CHIPERR_LOG_AND_THROW("PitchedDevPointer not found in allocation ranges",
                          hipErrorTbd);

  // Check if extents don't overextend the allocation?

  size_t Width = Extent.width;
  size_t Height = Extent.height;
  size_t Depth = Extent.depth;

  if (PitchedDevPtr.pitch == Extent.width) {
    return hipMemsetAsync(PitchedDevPtr.ptr, Value, Width * Height * Depth,
                          Stream);
  }

  // auto Height = std::max<size_t>(1, Extent.height);
  // auto Depth = std::max<size_t>(1, Extent.depth);
  auto Pitch = PitchedDevPtr.pitch;
  auto Dst = PitchedDevPtr.ptr;
  LOCK(ChipQueue->QueueMtx); // prevent interruptions
  hipError_t Res = hipSuccess;
  for (size_t i = 0; i < Depth; i++)
    for (size_t j = 0; j < Height; j++) {
      size_t SizeBytes = Width;
      auto Offset = i * (Pitch * PitchedDevPtr.ysize) + j * Pitch;
      char *DstP = (char *)Dst;
      auto Res = hipMemsetAsync(DstP + Offset, Value, SizeBytes, Stream);
      if (Res != hipSuccess)
        break;
    }

  RETURN(Res);
  CHIP_CATCH
}

hipError_t hipMemset3D(hipPitchedPtr PitchedDevPtr, int Value,
                       hipExtent Extent) {
  CHIP_TRY
  CHIPInitialize();

  auto ChipQueue = Backend->getActiveDevice()->getDefaultQueue();
  auto Res = hipMemset3DAsync(PitchedDevPtr, Value, Extent, ChipQueue);
  if (Res == hipSuccess)
    ChipQueue->finish();

  RETURN(Res);
  CHIP_CATCH
}

hipError_t hipMemsetAsync(void *Dst, int Value, size_t SizeBytes,
                          hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst);

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  const hipMemsetParams Params = {
      /* Dst */ Dst,
      /* elementSize*/ 1,
      /* height */ 1,
      /* pitch */ 1,
      /* value */ (unsigned int)Value, /* TODO Graphs - why is the arg for
                                          memset unsigned? */
      /* width */ SizeBytes};
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemset>(Params)) {
    RETURN(hipSuccess);
  }

  char CharVal = Value;
  ChipQueue->memFillAsync(Dst, SizeBytes, &CharVal, 1);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemset(void *Dst, int Value, size_t SizeBytes) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst);
  logInfo("hipMemset(Dst={}, Value={}, SizeBytes={})", Dst, Value, SizeBytes);

  char CharVal = Value;

  // Check if this pointer is registered
  auto AllocTracker = Backend->getActiveDevice()->AllocationTracker;
  auto AllocInfo = AllocTracker->getAllocInfo(Dst);

  if (!AllocInfo) {
    CHIPERR_LOG_AND_THROW("AllocInfo not found for the given pointer",
                          hipErrorInvalidValue);
  }

  Backend->getActiveDevice()->getDefaultQueue()->memFill(Dst, SizeBytes,
                                                         &CharVal, 1);

  if (AllocInfo->HostPtr) {
    logDebug("DevPtr {} is associated with HostPtr {}", AllocInfo->DevPtr,
             AllocInfo->HostPtr);
    if (AllocInfo->MemoryType == hipMemoryTypeUnified) {
      logDebug("AllocInfo->MemoryType == hipMemoryTypeUnified - skipping "
               "memset on host");
    } else if (AllocInfo->MemoryType == hipMemoryTypeHost) {
      logDebug("AllocInfo->MemoryType == hipMemoryTypeHost - executing memset "
               "on host");
      Backend->getActiveDevice()->getDefaultQueue()->MemMap(
          AllocInfo, CHIPQueue::MEM_MAP_TYPE::HOST_WRITE);
      memset(AllocInfo->HostPtr, Value, SizeBytes);
      Backend->getActiveDevice()->getDefaultQueue()->MemUnmap(AllocInfo);
    } else if (AllocInfo->MemoryType == hipMemoryTypeManaged) {
      UNIMPLEMENTED(hipErrorTbd);
    } else if (AllocInfo->MemoryType == hipMemoryTypeDevice) {
      CHIPERR_LOG_AND_THROW(
          "hipMemoryTypeDevice can't have an associated HostPtr", hipErrorTbd);
    } else if (AllocInfo->MemoryType == hipMemoryTypeArray) {
      CHIPERR_LOG_AND_THROW(
          "hipMemoryTypeArray can't have an associated HostPtr", hipErrorTbd);
    } else {
      CHIPERR_LOG_AND_THROW("Unknown MemoryType", hipErrorTbd);
    }
  }
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipMemsetD8Async(hipDeviceptr_t Dest, unsigned char Value,
                            size_t Count, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dest);

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  const hipMemsetParams Params = {
      /* Dst */ Dest,
      /* elementSize*/ 1,
      /* height */ 1,
      /* pitch */ 1,
      /* value */ (unsigned int)Value, /* TODO Graphs - why is the arg for
                                          memset unsigned? */
      /* width */ Count};
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemset>(Params)) {
    RETURN(hipSuccess);
  }

  ChipQueue->memFillAsync(Dest, 1 * Count, &Value, 1);
  RETURN(hipSuccess);

  CHIP_CATCH
};

hipError_t hipMemsetD8(hipDeviceptr_t Dest, unsigned char Value,
                       size_t SizeBytes) {
  RETURN(hipMemset(Dest, Value, SizeBytes));
}

hipError_t hipMemsetD16Async(hipDeviceptr_t Dest, unsigned short Value,
                             size_t Count, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  const hipMemsetParams Params = {
      /* Dst */ Dest,
      /* elementSize*/ 2,
      /* height */ 1,
      /* pitch */ 1,
      /* value */ (unsigned int)Value, /* TODO Graphs - why is the arg for
                                          memset unsigned? */
      /* width */ 2 * Count};
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemset>(Params)) {
    RETURN(hipSuccess);
  }

  ChipQueue->memFillAsync(Dest, 2 * Count, &Value, 2);
  RETURN(hipSuccess);

  CHIP_CATCH
}
hipError_t hipMemsetD16(hipDeviceptr_t Dest, unsigned short Value,
                        size_t Count) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dest);

  Backend->getActiveDevice()->getDefaultQueue()->memFill(Dest, 2 * Count,
                                                         &Value, 2);
  RETURN(hipSuccess);

  CHIP_CATCH
};

hipError_t hipMemsetD32Async(hipDeviceptr_t Dst, int Value, size_t Count,
                             hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  const hipMemsetParams Params = {
      /* Dst */ Dst,
      /* elementSize*/ 4,
      /* height */ 1,
      /* pitch */ 1,
      /* value */ (unsigned int)Value, /* TODO Graphs - why is the arg for
                                          memset unsigned? */
      /* width */ 4 * Count};
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemset>(Params)) {
    RETURN(hipSuccess);
  }

  ChipQueue->memFillAsync(Dst, 4 * Count, &Value, 4);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipMemsetD32(hipDeviceptr_t Dst, int Value, size_t Count) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst);

  Backend->getActiveDevice()->getDefaultQueue()->memFill(Dst, 4 * Count, &Value,
                                                         4);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipMemcpyParam2D(const hip_Memcpy2D *PCopy) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(PCopy);
  auto ChipQueue = Backend->getActiveDevice()->getDefaultQueue();
  auto Res = hipMemcpyParam2DAsync(PCopy, ChipQueue);
  if (Res == hipSuccess)
    ChipQueue->finish();

  RETURN(Res);
  CHIP_CATCH
}

hipError_t hipMemcpy2DAsync(void *Dst, size_t DPitch, const void *Src,
                            size_t SPitch, size_t Width, size_t Height,
                            hipMemcpyKind Kind, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Src);

  if (DPitch < 1)
    CHIPERR_LOG_AND_THROW("DPitch <= 0", hipErrorInvalidValue);
  if (SPitch < 1)
    CHIPERR_LOG_AND_THROW("SPitch <= 0", hipErrorInvalidValue);
  if (Width > DPitch)
    CHIPERR_LOG_AND_THROW("Width > DPitch", hipErrorInvalidValue);
  if (Height * Width == 0)
    return hipSuccess;

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  const hipMemcpy3DParms Params = {
      /* hipArray_t srcArray */ nullptr,
      /* struct hipPos srcPos */ make_hipPos(1, 1, 1),
      /* struct hipPitchedPtr srcPtr */
      make_hipPitchedPtr(const_cast<void *>(Src), SPitch, Width, Height),
      /* hipArray_t dstArray */ nullptr,
      /* struct hipPos dstPos */ make_hipPos(1, 1, 1),
      /* struct hipPitchedPtr dstPtr */
      make_hipPitchedPtr(Dst, SPitch, Width, Height),
      /* struct hipExtent extent */ make_hipExtent(Width, Height, 1),
      /* enum hipMemcpyKind kind */ Kind};
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemcpy>(Params)) {
    RETURN(hipSuccess);
  }

  if (SPitch == 0)
    SPitch = Width;
  if (DPitch == 0)
    DPitch = Width;

  if (SPitch == 0 || DPitch == 0)
    RETURN(hipErrorInvalidValue);
  LOCK(ChipQueue->QueueMtx); // prevent interruptions
  for (size_t i = 0; i < Height; ++i) {
    if (hipMemcpyAsync(Dst, Src, Width, Kind, Stream) != hipSuccess)
      RETURN(hipErrorLaunchFailure);
    Src = (char *)Src + SPitch;
    Dst = (char *)Dst + DPitch;
  }
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipMemcpy2D(void *Dst, size_t DPitch, const void *Src, size_t SPitch,
                       size_t Width, size_t Height, hipMemcpyKind Kind) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Src);
  if (SPitch < 1 || DPitch < 1 || Width > DPitch) {
    CHIPERR_LOG_AND_THROW("Source Pitch less than 1", hipErrorInvalidValue);
  }

  auto ChipQueue = Backend->getActiveDevice()->getDefaultQueue();

  hipError_t Res = hipMemcpy2DAsync(Dst, DPitch, Src, SPitch, Width, Height,
                                    Kind, ChipQueue);

  if (Res == hipSuccess)
    ChipQueue->finish();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpy2DToArray(hipArray *Dst, size_t WOffset, size_t HOffset,
                              const void *Src, size_t SPitch, size_t Width,
                              size_t Height, hipMemcpyKind Kind) {
  auto ChipQueue = Backend->getActiveDevice()->getDefaultQueue();

  auto Res = hipMemcpy2DToArrayAsync(Dst, WOffset, HOffset, Src, SPitch, Width,
                                     Height, Kind, ChipQueue);

  if (Res == hipSuccess)
    ChipQueue->finish();

  RETURN(Res);
}

hipError_t hipMemcpy2DToArrayAsync(hipArray *Dst, size_t WOffset,
                                   size_t HOffset, const void *Src,
                                   size_t SPitch, size_t Width, size_t Height,
                                   hipMemcpyKind Kind, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Src);

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  const hipMemcpy3DParms Params = {
      /* hipArray_t srcArray */ nullptr,
      /* struct hipPos srcPos */ make_hipPos(1, 1, 1),
      /* struct hipPitchedPtr srcPtr */
      make_hipPitchedPtr(const_cast<void *>(Src), SPitch, Width, Height),
      /* hipArray_t dstArray */ Dst,
      /* struct hipPos dstPos */ make_hipPos(WOffset, HOffset, 1),
      /* struct hipPitchedPtr dstPtr */ make_hipPitchedPtr(nullptr, 0, 0, 0),
      /* struct hipExtent extent */ make_hipExtent(Width, Height, 1),
      /* enum hipMemcpyKind kind */ Kind};
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemcpy>(Params)) {
    RETURN(hipSuccess);
  }

  if (!Dst)
    RETURN(hipErrorUnknown);

  size_t ByteSize = getChannelByteSize(Dst->desc);

  if ((WOffset + Width > (Dst->width * ByteSize)) || Width > SPitch)
    RETURN(hipErrorInvalidValue);

  size_t SrcW = SPitch;
  size_t DstW = (Dst->width) * ByteSize;
  LOCK(ChipQueue->QueueMtx); // prevent interruptions
  for (size_t Offset = HOffset; Offset < Height; ++Offset) {
    void *DstP = ((unsigned char *)Dst->data + Offset * DstW);
    void *SrcP = ((unsigned char *)Src + Offset * SrcW);
    if (hipMemcpyAsync(DstP, SrcP, Width, Kind, Stream) != hipSuccess)
      RETURN(hipErrorLaunchFailure);
  }

  RETURN(hipSuccess);
  CHIP_CATCH
};

hipError_t hipMemcpy2DFromArray(void *Dst, size_t DPitch, hipArray_const_t Src,
                                size_t WOffset, size_t HOffset, size_t Width,
                                size_t Height, hipMemcpyKind Kind) {
  auto ChipQueue = Backend->getActiveDevice()->getDefaultQueue();

  auto Res = hipMemcpy2DFromArrayAsync(Dst, DPitch, Src, WOffset, HOffset,
                                       Width, Height, Kind, ChipQueue);
  if (Res == hipSuccess)
    ChipQueue->finish();

  RETURN(Res);
}
hipError_t hipMemcpy2DFromArrayAsync(void *Dst, size_t DPitch,
                                     hipArray_const_t Src, size_t WOffset,
                                     size_t HOffset, size_t Width,
                                     size_t Height, hipMemcpyKind Kind,
                                     hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Src);
  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  const hipMemcpy3DParms Params = {
      /* hipArray_t srcArray */ const_cast<hipArray_t>(Src),
      /* struct hipPos srcPos */ make_hipPos(WOffset, HOffset, 1),
      /* struct hipPitchedPtr srcPtr */ make_hipPitchedPtr(nullptr, 0, 0, 0),
      /* hipArray_t dstArray */ nullptr,
      /* struct hipPos dstPos */ make_hipPos(1, 1, 1),
      /* struct hipPitchedPtr dstPtr */
      make_hipPitchedPtr(Dst, DPitch, Width, Height),
      /* struct hipExtent extent */ make_hipExtent(Width, Height, 1),
      /* enum hipMemcpyKind kind */ Kind};
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemcpy>(Params)) {
    RETURN(hipSuccess);
  }

  if (!Width || !Height)
    RETURN(hipSuccess);

  size_t ByteSize;
  if (Src) {
    switch (Src[0].desc.f) {
    case hipChannelFormatKindSigned:
      ByteSize = sizeof(int);
      break;
    case hipChannelFormatKindUnsigned:
      ByteSize = sizeof(unsigned int);
      break;
    case hipChannelFormatKindFloat:
      ByteSize = sizeof(float);
      break;
    case hipChannelFormatKindNone:
      ByteSize = sizeof(size_t);
      break;
    }
  } else {
    RETURN(hipErrorUnknown);
  }

  if ((WOffset + Width > (Src->width * ByteSize)) || Width > DPitch) {
    RETURN(hipErrorInvalidValue);
  }

  size_t DstW = DPitch;
  size_t SrcW = (Src->width) * ByteSize;
  LOCK(ChipQueue->QueueMtx); // prevent interruptions
  for (size_t Offset = 0; Offset < Height; ++Offset) {
    void *SrcP = ((unsigned char *)Src->data + Offset * SrcW);
    void *DstP = ((unsigned char *)Dst + Offset * DstW);
    auto Err = hipMemcpyAsync(DstP, SrcP, Width, Kind, Stream);
    ERROR_IF(Err != hipSuccess, Err);
  }

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyToArray(hipArray *Dst, size_t WOffset, size_t HOffset,
                            const void *Src, size_t Count, hipMemcpyKind Kind) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Src);

  void *DstP = (unsigned char *)Dst->data + WOffset;
  RETURN(hipMemcpy(DstP, Src, Count, Kind));
  CHIP_CATCH
}

hipError_t hipMemcpyFromArray(void *Dst, hipArray_const_t SrcArray,
                              size_t WOffset, size_t HOffset, size_t Count,
                              hipMemcpyKind Kind) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, SrcArray);

  void *SrcP = (unsigned char *)SrcArray->data + WOffset;
  RETURN(hipMemcpy(Dst, SrcP, Count, Kind));

  CHIP_CATCH
}

hipError_t hipMemcpyAtoH(void *Dst, hipArray *SrcArray, size_t SrcOffset,
                         size_t Count) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, SrcArray);
  if (SrcOffset > Count)
    CHIPERR_LOG_AND_THROW("Offset larger than count", hipErrorTbd);

  auto Info = Backend->getActiveDevice()->AllocationTracker->getAllocInfo(
      SrcArray->data);
  if (Info->Size < Count)
    CHIPERR_LOG_AND_THROW("MemCopy larger than allocated size", hipErrorTbd);

  return hipMemcpy((char *)Dst, (char *)SrcArray->data + SrcOffset, Count,
                   hipMemcpyDeviceToHost);

  CHIP_CATCH
}

hipError_t hipMemcpyHtoA(hipArray *DstArray, size_t DstOffset,
                         const void *SrcHost, size_t Count) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(SrcHost, DstArray);

  auto AllocTracker = Backend->getActiveDevice()->AllocationTracker;
  auto AllocInfo = AllocTracker->getAllocInfo(DstArray->data);
  if (!AllocInfo)
    CHIPERR_LOG_AND_THROW("Destination device pointer not allocated on device",
                          hipErrorTbd);
  if (DstOffset > AllocInfo->Size)
    CHIPERR_LOG_AND_THROW("Offset greater than allocation size", hipErrorTbd);
  if (Count > AllocInfo->Size)
    CHIPERR_LOG_AND_THROW("Copy size greater than allocation size",
                          hipErrorTbd);

  return hipMemcpy((char *)DstArray->data + DstOffset, SrcHost, Count,
                   hipMemcpyHostToDevice);

  CHIP_CATCH
}

hipError_t hipMemcpy3D(const struct hipMemcpy3DParms *Params) {
  CHIP_TRY
  CHIPInitialize();

  auto ChipQueue = Backend->getActiveDevice()->getDefaultQueue();
  auto Res = hipMemcpy3DAsync(Params, ChipQueue);
  if (Res == hipSuccess)
    ChipQueue->finish();

  RETURN(Res);
  CHIP_CATCH
}

hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms *Params,
                            hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Params);

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemcpy>(Params)) {
    RETURN(hipSuccess);
  }

  const HIP_MEMCPY3D PDrvI = getDrvMemcpy3DDesc(*Params);
  const HIP_MEMCPY3D *PDrv = &PDrvI;

  size_t ByteSize = 1;
  size_t Depth;
  size_t Height;
  size_t WidthInBytes;
  size_t SrcPitch;
  size_t DstPitch;
  void *SrcPtr;
  void *DstPtr;
  size_t YSize;

  bool ArrayDst = Params->dstArray != nullptr ? true : false;
  bool ArraySrc = Params->srcArray != nullptr ? true : false;
  bool DrvSrc = false;
  if (ArrayDst)
    DrvSrc = Params->dstArray->isDrv;

  if (ArrayDst || ArraySrc) {
    auto Desc = ArrayDst ? Params->dstArray->desc.f : Params->srcArray->desc.f;
    switch (Desc) {
    case hipChannelFormatKindSigned:
      ByteSize = sizeof(int);
      break;
    case hipChannelFormatKindUnsigned:
      ByteSize = sizeof(unsigned int);
      break;
    case hipChannelFormatKindFloat:
      ByteSize = sizeof(float);
      break;
    case hipChannelFormatKindNone:
      ByteSize = sizeof(size_t);
      break;
    }
  }

  if (ArrayDst && DrvSrc) {

    Depth = PDrv->Depth;
    Height = PDrv->Height;
    WidthInBytes = PDrv->WidthInBytes;
    DstPitch = PDrv->dstArray->width * 4;
    SrcPitch = PDrv->srcPitch;
    SrcPtr = (void *)PDrv->srcHost;
    YSize = PDrv->srcHeight;
    DstPtr = PDrv->dstArray->data;
  } else { // non Drc params
    Depth = Params->extent.depth;
    Height = Params->extent.height;
    WidthInBytes = Params->extent.width * ByteSize;

    if (ArraySrc) {
      SrcPitch = PDrv->srcArray->width * ByteSize; // ???
      SrcPtr = Params->srcArray->data;
    } else {
      SrcPitch = Params->srcPtr.pitch;
      SrcPtr = Params->srcPtr.ptr;
    }

    if (ArraySrc) {
      YSize = Params->srcArray->height;
    } else {
      YSize = Params->srcPtr.ysize;
    }

    if (ArrayDst) {
      DstPitch = Params->dstArray->width * ByteSize;
      DstPtr = Params->dstArray->data;
    } else {
      DstPitch = Params->dstPtr.pitch;
      DstPtr = Params->dstPtr.ptr;
    }
  }
  LOCK(ChipQueue->QueueMtx); // prevent interruptions
  if ((WidthInBytes == DstPitch) && (WidthInBytes == SrcPitch)) {
    return hipMemcpy((void *)DstPtr, (void *)SrcPtr,
                     WidthInBytes * Height * Depth, Params->kind);
  } else {
    for (size_t i = 0; i < Depth; i++) {
      for (size_t j = 0; j < Height; j++) {
        unsigned char *Src =
            (unsigned char *)SrcPtr + i * YSize * SrcPitch + j * SrcPitch;
        unsigned char *Dst =
            (unsigned char *)DstPtr + i * Height * DstPitch + j * DstPitch;
        if (hipMemcpyAsync(Dst, Src, WidthInBytes, Params->kind, Stream) !=
            hipSuccess)
          RETURN(hipErrorLaunchFailure);
      }
    }

    ChipQueue->finish();
    RETURN(hipSuccess);
  }
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipFuncGetAttributes(hipFuncAttributes *Attr,
                                const void *HostFunction) {
  CHIP_TRY
  CHIPInitialize();

  CHIPDevice *Dev = Backend->getActiveDevice();
  CHIPKernel *Kernel = Dev->findKernel(HostPtr(HostFunction));
  if (!Kernel)
    RETURN(hipErrorInvalidDeviceFunction);
  hipError_t Res = Kernel->getAttributes(Attr);
  RETURN(Res);

  CHIP_CATCH
}

hipError_t hipModuleGetGlobal(hipDeviceptr_t *Dptr, size_t *Bytes,
                              hipModule_t Hmod, const char *Name) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dptr, Bytes, Hmod, Name);
  auto ChipModule = static_cast<CHIPModule *>(Hmod);

  CHIPDeviceVar *Var = ChipModule->getGlobalVar(Name);
  *Dptr = Var->getDevAddr();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGetSymbolSize(size_t *Size, const void *Symbol) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Size, Symbol);

  CHIPDeviceVar *Var =
      Backend->getActiveDevice()->getGlobalVar((const char *)Symbol);
  ERROR_IF(!Var, hipErrorInvalidSymbol);

  *Size = Var->getSize();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyToSymbol(const void *Symbol, const void *Src,
                             size_t SizeBytes, size_t Offset,
                             hipMemcpyKind Kind) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Symbol, Src);
  auto ChipQueue = Backend->getActiveDevice()->getDefaultQueue();

  hipError_t Res =
      hipMemcpyToSymbolAsync(Symbol, Src, SizeBytes, Offset, Kind, ChipQueue);

  if (Res == hipSuccess)
    ChipQueue->finish();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyToSymbolAsync(const void *Symbol, const void *Src,
                                  size_t SizeBytes, size_t Offset,
                                  hipMemcpyKind Kind, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Symbol, Src);

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemcpyToSymbol>(
          const_cast<void *>(Src), Symbol, SizeBytes, Offset, Kind)) {
    RETURN(hipSuccess);
  }

  Backend->getActiveDevice()->prepareDeviceVariables(HostPtr(Symbol));

  CHIPDeviceVar *Var = Backend->getActiveDevice()->getGlobalVar(Symbol);
  ERROR_IF(!Var, hipErrorInvalidSymbol);
  void *DevPtr = Var->getDevAddr();
  assert(DevPtr && "Found the symbol but not its device address?");

  RETURN(hipMemcpyAsync((void *)((intptr_t)DevPtr + Offset), Src, SizeBytes,
                        Kind, Stream));
  CHIP_CATCH
}

hipError_t hipMemcpyFromSymbol(void *Dst, const void *Symbol, size_t SizeBytes,
                               size_t Offset, hipMemcpyKind Kind) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Symbol);

  auto ChipQueue = Backend->getActiveDevice()->getDefaultQueue();

  hipError_t Res =
      hipMemcpyFromSymbolAsync(Dst, Symbol, SizeBytes, Offset, Kind, ChipQueue);

  if (Res == hipSuccess)
    ChipQueue->finish();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyFromSymbolAsync(void *Dst, const void *Symbol,
                                    size_t SizeBytes, size_t Offset,
                                    hipMemcpyKind Kind, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Symbol);

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeMemcpyFromSymbol>(
          const_cast<void *>(Dst), Symbol, SizeBytes, Offset, Kind)) {
    RETURN(hipSuccess);
  }

  Backend->getActiveDevice()->prepareDeviceVariables(HostPtr(Symbol));
  CHIPDeviceVar *Var = ChipQueue->getDevice()->getGlobalVar(Symbol);
  ERROR_IF(!Var, hipErrorInvalidSymbol);
  void *DevPtr = Var->getDevAddr();

  RETURN(hipMemcpyAsync(Dst, (void *)((intptr_t)DevPtr + Offset), SizeBytes,
                        Kind, Stream));
  CHIP_CATCH
}

hipError_t hipModuleLoadData(hipModule_t *ModuleHandle, const void *Image) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(ModuleHandle, Image);

  std::string ErrorMsg;
  // Image is expected to be a Clang offload bundle.
  std::string_view ModuleCode = extractSPIRVModule(Image, ErrorMsg);
  if (ModuleCode.empty()) {
    logDebug("{}", ErrorMsg);
    RETURN(hipErrorTbd);
  }

  auto Entry = getSPVRegister().registerSource(ModuleCode);
  auto *SrcMod = getSPVRegister().getSource(Entry);
  auto *ChipModule = Backend->getActiveDevice()->getOrCreateModule(*SrcMod);
  *ModuleHandle = ChipModule;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleLoadDataEx(hipModule_t *Module, const void *Image,
                               unsigned int NumOptions, hipJitOption *Options,
                               void **OptionValues) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Module, Image);
  RETURN(hipModuleLoadData(Module, Image));
  CHIP_CATCH
}

hipError_t hipLaunchKernel(const void *HostFunction, dim3 GridDim,
                           dim3 BlockDim, void **Args, size_t SharedMem,
                           hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(HostFunction, Args);

  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  if (ChipQueue->captureIntoGraph<CHIPGraphNodeKernel>(
          HostFunction, GridDim, BlockDim, Args, SharedMem)) {
    RETURN(hipSuccess);
  }

  auto *Device = Backend->getActiveDevice();
  Device->prepareDeviceVariables(HostPtr(HostFunction));

  auto *ChipKernel = Device->findKernel(HostPtr(HostFunction));
  if (!ChipKernel)
    CHIPERR_LOG_AND_THROW("Unexpected error: could not find a kernel.",
                          hipErrorTbd);
  ChipQueue->launchKernel(ChipKernel, GridDim, BlockDim, Args, SharedMem);
  handleAbortRequest(*ChipQueue, *ChipKernel->getModule());

  RETURN(hipSuccess);
  CHIP_CATCH
}

static unsigned getNumTextureDimensions(const hipResourceDesc *ResDesc) {
  switch (ResDesc->resType) {
  default:
    CHIPASSERT(false && "Unknown resource type.");
    return 0;
  case hipResourceTypeLinear:
    return 1;
  case hipResourceTypePitch2D:
    return 2;
  case hipResourceTypeArray: {
    switch (ResDesc->res.array.array->textureType) {
    default:
      CHIPASSERT(false && "Unknown texture type.");
      return 0;
    case hipTextureType1D:
    case hipTextureType1DLayered:
      return 1;
    case hipTextureType2D:
    case hipTextureType2DLayered:
    case hipTextureTypeCubemap:
    case hipTextureTypeCubemapLayered:
      return 2;
    case hipTextureType3D:
      return 3;
    }
  }
  }
  CHIPASSERT(false && "Unreachable.");
  return 0;
}

hipError_t
hipCreateTextureObject(hipTextureObject_t *TexObject,
                       const hipResourceDesc *ResDesc,
                       const hipTextureDesc *TexDesc,
                       const struct hipResourceViewDesc *ResViewDesc) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(TexObject, ResDesc, TexDesc);

  // Check the descriptions are valid and supported.
  switch (ResDesc->resType) {
  default:
    RETURN(hipErrorInvalidValue);
  case hipResourceTypeArray: {
    if (!ResDesc->res.array.array || !ResDesc->res.array.array->data)
      RETURN(hipErrorInvalidValue);

    break;
  }
  case hipResourceTypeLinear: {
    if (!ResDesc->res.linear.devPtr)
      RETURN(hipErrorInvalidValue);

    size_t MaxTexInTexels = Backend->getActiveDevice()->getAttr(
        hipDeviceAttributeMaxTexture1DLinear);
    size_t MaxTexInBytes =
        MaxTexInTexels * getChannelByteSize(ResDesc->res.linear.desc);
    if (ResDesc->res.linear.sizeInBytes > MaxTexInBytes)
      RETURN(hipErrorInvalidValue);

    break;
  }
  case hipResourceTypePitch2D: {
    auto &Pitch2dDesc = ResDesc->res.pitch2D;
    if (!Pitch2dDesc.devPtr)
      RETURN(hipErrorInvalidValue);

    size_t PitchInTexels =
        Pitch2dDesc.pitchInBytes / getChannelByteSize(Pitch2dDesc.desc);
    if (PitchInTexels < Pitch2dDesc.width)
      RETURN(hipErrorInvalidValue);

    size_t MaxDimSize = Backend->getActiveDevice()->getAttr(
        hipDeviceAttributeMaxTexture2DLinear);
    if (Pitch2dDesc.width > MaxDimSize || Pitch2dDesc.height > MaxDimSize ||
        PitchInTexels > MaxDimSize)
      RETURN(hipErrorInvalidValue);

    break;
  }
  };

  unsigned NumDims = getNumTextureDimensions(ResDesc);
  bool AddrModeSupported =
      (NumDims < 2 || TexDesc->addressMode[0] == TexDesc->addressMode[1]) &&
      (NumDims < 3 || TexDesc->addressMode[0] == TexDesc->addressMode[2]);
  if (!AddrModeSupported)
    CHIPERR_LOG_AND_THROW(
        "Heterogeneous texture addressing modes are not supported yet",
        hipErrorTbd);

  CHIPTexture *RetObj =
      Backend->getActiveDevice()->createTexture(ResDesc, TexDesc, ResViewDesc);
  if (RetObj != nullptr) {
    *TexObject = reinterpret_cast<hipTextureObject_t>(RetObj);
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);
  CHIP_CATCH
}

hipError_t hipDestroyTextureObject(hipTextureObject_t TextureObject) {
  CHIP_TRY
  CHIPInitialize();
  // TODO CRITCAL look into the define for hipTextureObject_t
  if (TextureObject == nullptr)
    RETURN(hipSuccess);
  CHIPTexture *ChipTexture = (CHIPTexture *)TextureObject;
  Backend->getActiveDevice()->destroyTexture(ChipTexture);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc *ResDesc,
                                           hipTextureObject_t TextureObject) {
  CHIP_TRY
  CHIPInitialize();
  if (TextureObject == nullptr)
    RETURN(hipErrorInvalidValue);
  CHIPTexture *ChipTexture = (CHIPTexture *)TextureObject;
  *ResDesc = ChipTexture->getResourceDesc();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleLoad(hipModule_t *Module, const char *FuncName) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Module, FuncName);

#if 0
  // TODO: This is likely bit-rotted (due to lack of testing).
  //       Reimplement this again.

  std::ifstream ModuleFile(FuncName,
                           std::ios::in | std::ios::binary | std::ios::ate);
  ERROR_IF((ModuleFile.fail()), hipErrorFileNotFound);

  size_t Size = ModuleFile.tellg();
  char *MemBlock = new char[Size];
  ModuleFile.seekg(0, std::ios::beg);
  ModuleFile.read(MemBlock, Size);
  ModuleFile.close();
  std::string Content(MemBlock, Size);
  delete[] MemBlock;

  // CHIPModule *chip_module = new CHIPModule(std::move(content));
  for (auto &Dev : Backend->getDevices())
    Dev->addModule(&Content);
#endif
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleUnload(hipModule_t Module) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Module);
  logInfo("hipModuleUnload(Module={}", (void *)Module);

  auto *ChipModule = reinterpret_cast<CHIPModule *>(Module);
  const auto &SrcMod = ChipModule->getSourceModule();
  Backend->getActiveDevice()->eraseModule(ChipModule);
  getSPVRegister().unregisterSource(&SrcMod);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleGetFunction(hipFunction_t *Function, hipModule_t Module,
                                const char *Name) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Function, Module, Name);
  auto ChipModule = (CHIPModule *)Module;
  CHIPKernel *Kernel = ChipModule->getKernelByName(Name);

  ERROR_IF((Kernel == nullptr), hipErrorInvalidDeviceFunction);

  *Function = Kernel;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleLaunchKernel(hipFunction_t Kernel, unsigned int GridDimX,
                                 unsigned int GridDimY, unsigned int GridDimZ,
                                 unsigned int BlockDimX, unsigned int BlockDimY,
                                 unsigned int BlockDimZ,
                                 unsigned int SharedMemBytes,
                                 hipStream_t Stream, void *KernelParams[],
                                 void *Extra[]) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));

  if (KernelParams == Extra)
    CHIPERR_LOG_AND_THROW("either kernelParams or extra is required",
                          hipErrorLaunchFailure);

  dim3 Grid(GridDimX, GridDimY, GridDimZ);
  dim3 Block(BlockDimX, BlockDimY, BlockDimZ);

  auto ChipKernel = static_cast<CHIPKernel *>(Kernel);
  Backend->getActiveDevice()->prepareDeviceVariables(
      HostPtr(ChipKernel->getHostPtr()));

  if (KernelParams)
    ChipQueue->launchKernel(ChipKernel, Grid, Block, KernelParams,
                            SharedMemBytes);
  else {
    // Convert the "extra" argument passing style to KernelParams's
    // format (an array of pointers to the argument data) for avoiding
    // adding another argument processing logic in the downstream.

    void *ExtraArgBuf = nullptr;
    // Some limit to avoid a run away case (e.g. missing HIP_LAUNCH_PARAM_END).
    constexpr unsigned ArgLimit = 100;
    for (unsigned i = 0; Extra[i] != HIP_LAUNCH_PARAM_END && i < ArgLimit;
         i++) {
      if (Extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER)
        ExtraArgBuf = Extra[++i];
      else if (Extra[i] == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
        i++; // Ignore setting value.
        continue;
      } else
        RETURN(hipErrorInvalidValue);
    }

    if (!ExtraArgBuf) // Null argument pointer.
      RETURN(hipErrorInvalidValue);

    auto ChipKernel = static_cast<CHIPKernel *>(Kernel);

    auto *FuncInfo = ChipKernel->getFuncInfo();
    auto ParamBuffer = convertExtraArgsToPointerArray(ExtraArgBuf, *FuncInfo);

    ChipQueue->launchKernel(ChipKernel, Grid, Block, ParamBuffer.data(),
                            SharedMemBytes);
  }

  handleAbortRequest(*ChipQueue, *ChipKernel->getModule());
  return hipSuccess;
  CHIP_CATCH
}

hipError_t hipExtModuleLaunchKernel(
    hipFunction_t Kernel,
    // NOTE: Grid units are threads/work-items instead of blocks/workgroups.
    uint32_t GlobalWorkSizeX, uint32_t GlobalWorkSizeY,
    uint32_t GlobalWorkSizeZ, uint32_t LocalWorkSizeX, uint32_t LocalWorkSizeY,
    uint32_t LocalWorkSizeZ, size_t SharedMemBytes, hipStream_t Stream,
    void **KernelParams, void **Extra, hipEvent_t StartEvent,
    hipEvent_t StopEvent, uint32_t Flags) {

  CHIP_TRY
  NULLCHECK(Kernel);
  // Null checks on the KernelParams and Extra arguments are performed by
  // hipModuleLaunchKernel().
  CHIPInitialize();

  // TODO: Process flags (hipExtAnyOrderLaunch).

  // Check local sizes divide grids.
  if (GlobalWorkSizeX % LocalWorkSizeX != 0 ||
      GlobalWorkSizeY % LocalWorkSizeY != 0 ||
      GlobalWorkSizeZ % LocalWorkSizeZ != 0)
    RETURN(hipErrorInvalidValue);

  auto GridBlocksX = GlobalWorkSizeX / LocalWorkSizeX;
  auto GridBlocksY = GlobalWorkSizeY / LocalWorkSizeY;
  auto GridBlocksZ = GlobalWorkSizeZ / LocalWorkSizeZ;

  hipError_t Result = hipSuccess;

  if (StartEvent)
    Result = hipEventRecord(StartEvent, Stream);
  if (Result != hipSuccess)
    RETURN(Result);

  Result = hipModuleLaunchKernel(Kernel, GridBlocksX, GridBlocksY, GridBlocksZ,
                                 LocalWorkSizeX, LocalWorkSizeY, LocalWorkSizeZ,
                                 SharedMemBytes, Stream, KernelParams, Extra);
  if (Result != hipSuccess)
    RETURN(Result);

  if (StopEvent)
    Result = hipEventRecord(StopEvent, Stream);

  RETURN(Result);
  CHIP_CATCH
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

hipError_t hipLaunchByPtr(const void *HostFunction) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(HostFunction);

  logTrace("hipLaunchByPtr");
  Backend->getActiveDevice()->prepareDeviceVariables(HostPtr(HostFunction));
  CHIPExecItem *ExecItem = ChipExecStack.top();
  ChipExecStack.pop();

  auto ChipQueue = ExecItem->getQueue();
  if (!ChipQueue) {
    std::string Msg = "Tried to launch CHIPExecItem but its queue is null";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  }

  auto *ChipDev = ChipQueue->getDevice();
  auto *ChipKernel = ChipDev->findKernel(HostPtr(HostFunction));
  ExecItem->setKernel(ChipKernel);

  ChipQueue->launch(ExecItem);
  handleAbortRequest(*ChipQueue, *ChipKernel->getModule());
  delete ExecItem;

  return hipSuccess;
  CHIP_CATCH
}

hipError_t hipConfigureCall(dim3 GridDim, dim3 BlockDim, size_t SharedMem,
                            hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipQueue = Backend->findQueue(static_cast<CHIPQueue *>(Stream));
  logDebug("hipConfigureCall()");
  RETURN(Backend->configureCall(GridDim, BlockDim, SharedMem, ChipQueue));
  RETURN(hipSuccess);
  CHIP_CATCH
}

// Reference of HIP entity registration, generated by Clang, presented
// in pseudo C++ which was derived from LLVM bitcode:
//
//   static void **Handle = nullptr;
//
//   static void __attribute__((constructor)) __hip_module_ctor() {
//     if (!Handle)
//       Handle = __hipRegisterFatBinary(...);
//     __hip_register_globals(Handle);
//     atexit(__hip_module_dtor);
//   }
//
//   static void __hip_register_globals(void **Handle) {
//     // One for each __global__ definition.
//     __hipRegisterFunction(Handle, ...);
//     ...
//     // One for each __device__ and __constant__ variable in the module.
//     __hipRegisterVar(Handle, ...);
//     ...
//     // And possibly more registrations of other entities (legacy
//     // textures, surfaces, managed variables, etc.)
//   }
//
//   static void __hip_module_dtor(void **Handle) {
//     if (Handle) {
//       __hipUnregisterFatBinary(Handle);
//       Handle = nullptr;
//     }
//   }

extern "C" void **__hipRegisterFatBinary(const void *Data) {
  CHIP_TRY
  // Increment before calling potentially exception throwing functions.
  CHIPNumRegisteredFatBinaries++;

  logDebug("__hipRegisterFatBinary");

  // NOTE: CHIP backend initialization is undesired here. This is done
  //       for avoiding start-up lag and other unexpected issues that
  //       may come from the backend before a client makes any HIP API
  //       function call.

  // FIXME: There are segfaults that occur sometimes at program exit
  //        in some cases (e.g. in Unit_hipStreamPerThread_DeviceReset_1 test
  //        case) and they go away if we have the CHIP runtime initialized
  //        early. Should find the causes, fix them and then and then remove the
  //        CHIPInitialize() call.
  CHIPInitialize();

  const __CudaFatBinaryWrapper *Wrapper =
      reinterpret_cast<const __CudaFatBinaryWrapper *>(Data);
  if (Wrapper->magic != __hipFatMAGIC2 || Wrapper->version != 1) {
    CHIPERR_LOG_AND_THROW("The given object is not hipFatBinary",
                          hipErrorInitializationError);
  }

  std::string ErrorMsg;
  auto SPIRVModuleSpan = extractSPIRVModule(Wrapper->binary, ErrorMsg);
  if (SPIRVModuleSpan.empty())
    CHIPERR_LOG_AND_THROW(ErrorMsg, hipErrorInitializationError);

  auto ModHandle = getSPVRegister().registerSource(SPIRVModuleSpan);
  logDebug("Registered SPIR-V module {}, source-binary={}",
           static_cast<const void *>(ModHandle.Module),
           static_cast<const void *>(SPIRVModuleSpan.data()));
  return (void **)ModHandle.Module;

  CHIP_CATCH_NO_RETURN
  return nullptr;
}

extern "C" void __hipUnregisterFatBinary(void *Data) {
  // With current Clang codegen this method is not called with nullptr.
  assert(Data && "__hipUnregisterFatBinary called with nullptr!");

  CHIP_TRY

  // Decrement before calling potentially exception throwing functions.
  auto NumBins = CHIPNumRegisteredFatBinaries.fetch_sub(1);
  assert(NumBins && "Underflow!");

  logDebug("Unregister module: {}", Data);
  SPVRegister::Handle ModHandle{Data};
  getSPVRegister().unregisterSource(ModHandle);

  logDebug("Modules left: {}", NumBins - 1);

  if (Backend && NumBins == 1)
    CHIPUninitialize();

  CHIP_CATCH_NO_RETURN
}

extern "C" void __hipRegisterFunction(void **Data, const void *HostFunction,
                                      char *DeviceFunction,
                                      const char *DeviceName,
                                      unsigned int ThreadLimit, void *Tid,
                                      void *Bid, dim3 *BlockDim, dim3 *GridDim,
                                      int *WSize) {
  if (!Data) // May happen if the fatbin registration failed.
    return;

  CHIP_TRY
  // NOTE: CHIP backend initialization is undesired here. See the
  //       rationale in __hipRegisterFatBinary().

  logDebug("Module {}: register function ({}) {}",
           static_cast<const void *>(Data),
           static_cast<const void *>(HostFunction), DeviceName);
  SPVRegister::Handle ModHandle{reinterpret_cast<void *>(Data)};
  getSPVRegister().bindFunction(ModHandle, HostPtr(HostFunction), DeviceName);
  CHIP_CATCH_NO_RETURN
}

hipError_t hipSetupArgument(const void *Arg, size_t Size, size_t Offset) {
  CHIP_TRY
  CHIPInitialize();
  // Development focus has been on the new HIP launch API so this path
  // is likely bitrotted.
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

extern "C" void
__hipRegisterVar(void **Data,
                 void *Var,        // The shadow variable in host code
                 char *HostName,   // Variable name in host code
                 char *DeviceName, // Variable name in device code
                 int Ext,          // Whether this variable is external
                 int Size,         // Size of the variable
                 int Constant,     // Whether this variable is constant
                 int Global        // Unknown, always 0
) {
  assert(Ext == 0);    // Device code should be fully linked so no
                       // external variables.
  assert(Global == 0); // HIP-Clang fixes this to zero.
  assert(std::string(HostName) == std::string(DeviceName));

  if (!Data) // May happen if the fatbin registration failed.
    return;

  CHIP_TRY
  // NOTE: CHIP backend initialization is undesired here. See the
  //       rationale in __hipRegisterFatBinary().

  logDebug("Module {}: Register variable ({}) size={}, name={}", (void *)Data,
           (void *)Var, Size, DeviceName);

  SPVRegister::Handle ModHandle{reinterpret_cast<void *>(Data)};
  getSPVRegister().bindVariable(ModHandle, HostPtr(Var), DeviceName, Size);

  CHIP_CATCH_NO_RETURN
}

/*
 *
__hipRegisterTexture (void **fatCubinHandle,
                       const struct textureReference *hostVar, // shadow
variable in host code const void **deviceAddress, // actually variable name
                       const char *deviceName, // variable name, same as ^^
                       int TextureType, // 1D/2D/3D
                       int Normalized, //
                       int Extern)
*/

hipError_t hipGetSymbolAddress(void **DevPtr, const void *Symbol) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DevPtr, Symbol);

  Backend->getActiveDevice()->prepareDeviceVariables(HostPtr(Symbol));
  CHIPDeviceVar *Var = Backend->getActiveDevice()->getGlobalVar(Symbol);
  ERROR_IF(!Var, hipErrorInvalidSymbol);
  *DevPtr = Var->getDevAddr();
  assert(*DevPtr);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipIpcOpenEventHandle(hipEvent_t *Event,
                                 hipIpcEventHandle_t Handle) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t *Handle, hipEvent_t Event) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

hipError_t hipModuleOccupancyMaxPotentialBlockSize(int *GridSize,
                                                   int *BlockSize,
                                                   hipFunction_t Func,
                                                   size_t DynSharedMemPerBlk,
                                                   int BlockSizeLimit);

hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
    int *GridSize, int *BlockSize, hipFunction_t Func,
    size_t DynSharedMemPerBlk, int BlockSizeLimit, unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
    int *NumBlocks, hipFunction_t Func, int BlockSize,
    size_t DynSharedMemPerBlk) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *NumBlocks, hipFunction_t Func, int BlockSize,
    size_t DynSharedMemPerBlk, unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

hipError_t
hipOccupancyMaxActiveBlocksPerMultiprocessor(int *NumBlocks, const void *Func,
                                             int BlockSize,
                                             size_t DynSharedMemPerBlk) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *NumBlocks, const void *Func, int BlockSize, size_t DynSharedMemPerBlk,
    unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

hipError_t hipOccupancyMaxPotentialBlockSize(int *GridSize, int *BlockSize,
                                             const void *Func,
                                             size_t DynSharedMemPerBlk,
                                             int BlockSizeLimit) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

hipError_t hipGetDeviceFlags(unsigned int *Flags) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

/************************************************************
 ************************************************************
 ************************************************************/

// returning a hipError_t from these API is problematic, because icpx is used
// as compiler for sycl and the use of hipError_t mandates inclusion
// of hip/hip_runtime.h which is not compatible which icpx

int hipGetBackendNativeHandles(uintptr_t Stream, uintptr_t *NativeHandles,
                               int *NumHandles) {
  CHIP_TRY
  CHIPInitialize();
  logDebug("hipGetBackendNativeHandles");
  auto ChipQueue = Backend->findQueue(reinterpret_cast<CHIPQueue *>(Stream));
  RETURN(ChipQueue->getBackendHandles(NativeHandles, NumHandles));
  CHIP_CATCH
}

int hipInitFromNativeHandles(const uintptr_t *NativeHandles, int NumHandles) {
  CHIP_TRY
  logDebug("hipInitFromNativeHandles");
  RETURN(CHIPReinitialize(NativeHandles, NumHandles));
  CHIP_CATCH
}

void *hipGetNativeEventFromHipEvent(void *HipEvent) {
  logDebug("hipGetNativeEventFromHipEvent");
  void *E = nullptr;
  CHIP_TRY
  CHIPInitialize();

  if (HipEvent == NULL)
    return NULL;

  E = Backend->getNativeEvent((hipEvent_t)HipEvent);
  CHIP_CATCH_NO_RETURN
  return E;
}

void *hipGetHipEventFromNativeEvent(void *NativeEvent) {
  logDebug("hipGetHipEventFromNativeEvent");
  hipEvent_t E = nullptr;
  CHIP_TRY
  CHIPInitialize();

  if (NativeEvent == NULL)
    return NULL;

  E = Backend->getHipEvent(NativeEvent);
  CHIP_CATCH_NO_RETURN
  return E;
}

const char *hipGetBackendName() {
  logDebug("hipGetCHIPBackend");
  return CHIPGetBackendName();
}

hipError_t hipProfilerStart() {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

hipError_t hipProfilerStop() {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

#endif
