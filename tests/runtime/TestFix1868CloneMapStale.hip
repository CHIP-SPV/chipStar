// Reproduces: the original -> clone node map of a cloned or instantiated
// graph is never updated, so lookups return destroyed clone nodes, and a new
// node allocated at a destroyed original node's address resolves to the
// destroyed node's clone.
#include <hip/hip_runtime_api.h>
#include <iostream>

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Add memset (or empty) nodes to Graph until one lands at Old's address.
static hipGraphNode_t addAt(hipGraph_t Graph, hipGraphNode_t Old,
                            const hipMemsetParams *Memset) {
  for (int I = 0; I < 256; I++) {
    hipGraphNode_t Node;
    hipError_t Err = Memset ? hipGraphAddMemsetNode(&Node, Graph, nullptr, 0,
                                                    Memset)
                            : hipGraphAddEmptyNode(&Node, Graph, nullptr, 0);
    if (Err != hipSuccess)
      return nullptr;
    if (Node == Old)
      return Node;
  }
  return nullptr;
}

int main() {
  int Failed = 0;

  // A destroyed node of a cloned graph must no longer be found in it.
  hipGraph_t Graph, Clone;
  hipGraphNode_t A, InClone;
  CHECK(hipGraphCreate(&Graph, 0));
  CHECK(hipGraphAddEmptyNode(&A, Graph, nullptr, 0));
  CHECK(hipGraphClone(&Clone, Graph));
  CHECK(hipGraphNodeFindInClone(&InClone, A, Clone));
  // A node of one clone is not a node of the graph another clone came from.
  hipGraph_t Sibling;
  hipGraphNode_t InSibling;
  CHECK(hipGraphClone(&Sibling, Graph));
  if (hipGraphNodeFindInClone(&InSibling, InClone, Sibling) == hipSuccess) {
    std::cout << "FAIL: hipGraphNodeFindInClone found a sibling clone's node\n";
    Failed = 1;
  }
  CHECK(hipGraphDestroy(Sibling));
  CHECK(hipGraphDestroyNode(InClone));
  if (hipGraphNodeFindInClone(&InClone, A, Clone) == hipSuccess) {
    std::cout << "FAIL: hipGraphNodeFindInClone returned a destroyed node\n";
    Failed = 1;
  }
  CHECK(hipGraphDestroy(Clone));

  // A new node at the address of a destroyed original node was not part of
  // the graph when it was cloned.
  CHECK(hipGraphClone(&Clone, Graph));
  CHECK(hipGraphDestroyNode(A));
  // Without address reuse (e.g. under a sanitizer) there is nothing to check.
  hipGraphNode_t B = addAt(Graph, A, nullptr);
  if (B && hipGraphNodeFindInClone(&InClone, B, Clone) == hipSuccess) {
    std::cout << "FAIL: hipGraphNodeFindInClone found a node added after "
                 "the clone\n";
    Failed = 1;
  }
  CHECK(hipGraphDestroy(Clone));
  CHECK(hipGraphDestroy(Graph));

  // The same for an executable graph: a node added after instantiation at a
  // destroyed node's address has no executable copy.
  void *Dev;
  CHECK(hipMalloc(&Dev, 64));
  hipMemsetParams Params = {};
  Params.dst = Dev;
  Params.elementSize = 1;
  Params.width = 64;
  Params.height = 1;
  hipGraphNode_t M;
  hipGraphExec_t Exec;
  CHECK(hipGraphCreate(&Graph, 0));
  CHECK(hipGraphAddMemsetNode(&M, Graph, nullptr, 0, &Params));
  CHECK(hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0));
  CHECK(hipGraphDestroyNode(M));
  hipGraphNode_t M2 = addAt(Graph, M, &Params);
  unsigned int Enabled = 0;
  if (M2 && hipGraphNodeGetEnabled(Exec, M2, &Enabled) == hipSuccess) {
    std::cout << "FAIL: hipGraphNodeGetEnabled found a node added after "
                 "instantiation\n";
    Failed = 1;
  }
  CHECK(hipGraphExecDestroy(Exec));
  CHECK(hipGraphDestroy(Graph));
  CHECK(hipFree(Dev));

  std::cout << (Failed ? "FAILED\n" : "PASSED\n");
  return Failed;
}
