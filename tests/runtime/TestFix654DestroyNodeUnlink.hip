// Reproduces #654: destroying a node removes it and its edges from its graph.
#include <hip/hip_runtime_api.h>
#include <cstdio>

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t Err = (x);                                                      \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %d (%s)\n", #x, Err, hipGetErrorName(Err));    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  hipGraph_t Graph;
  CHECK(hipGraphCreate(&Graph, 0));
  hipGraphNode_t A, B;
  CHECK(hipGraphAddEmptyNode(&A, Graph, nullptr, 0));
  CHECK(hipGraphAddEmptyNode(&B, Graph, &A, 1));

  hipGraph_t Clone;
  CHECK(hipGraphClone(&Clone, Graph));
  hipGraphNode_t CloneA, CloneB;
  CHECK(hipGraphNodeFindInClone(&CloneA, A, Clone));
  CHECK(hipGraphNodeFindInClone(&CloneB, B, Clone));
  CHECK(hipGraphDestroyNode(CloneA));
  size_t CloneNodes = 0, CloneDeps = 0;
  CHECK(hipGraphGetNodes(Clone, nullptr, &CloneNodes));
  CHECK(hipGraphNodeGetDependencies(CloneB, nullptr, &CloneDeps));
  if (CloneNodes != 1 || CloneDeps != 0) {
    printf("FAIL: clone has %zu nodes (expected 1), %zu dependencies "
           "(expected 0)\n",
           CloneNodes, CloneDeps);
    return 1;
  }
  CHECK(hipGraphDestroy(Clone));

  CHECK(hipGraphDestroyNode(A));

  size_t NumNodes = 0, NumDeps = 0;
  CHECK(hipGraphGetNodes(Graph, nullptr, &NumNodes));
  CHECK(hipGraphNodeGetDependencies(B, nullptr, &NumDeps));
  if (NumNodes != 1 || NumDeps != 0) {
    printf("FAIL: %zu nodes (expected 1), %zu dependencies (expected 0)\n",
           NumNodes, NumDeps);
    return 1;
  }

  hipGraphExec_t Exec;
  CHECK(hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(Exec, nullptr));
  CHECK(hipStreamSynchronize(nullptr));
  CHECK(hipGraphExecDestroy(Exec));
  CHECK(hipGraphDestroy(Graph));

  // Destroying a node must also clear reverse edges left by edge removal.
  hipGraph_t G2;
  CHECK(hipGraphCreate(&G2, 0));
  hipGraphNode_t C, D;
  CHECK(hipGraphAddEmptyNode(&C, G2, nullptr, 0));
  CHECK(hipGraphAddEmptyNode(&D, G2, &C, 1));
  CHECK(hipGraphRemoveDependencies(G2, &C, &D, 1));
  CHECK(hipGraphDestroyNode(D));
  size_t Count = 1;
  CHECK(hipGraphNodeGetDependentNodes(C, nullptr, &Count));
  if (Count != 0) {
    printf("FAIL: C has %zu dependent nodes (expected 0)\n", Count);
    return 1;
  }
  CHECK(hipGraphDestroyNode(C));
  size_t N2 = 1;
  CHECK(hipGraphGetNodes(G2, nullptr, &N2));
  if (N2 != 0) {
    printf("FAIL: %zu nodes (expected 0)\n", N2);
    return 1;
  }
  CHECK(hipGraphDestroy(G2));
  printf("PASSED\n");
  return 0;
}
