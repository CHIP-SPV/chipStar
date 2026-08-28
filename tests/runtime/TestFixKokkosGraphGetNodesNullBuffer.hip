// Reproduces: hipGraphGetNodes and hipGraphGetRootNodes write through the
// output array even when it is null, although a null array is how callers
// ask for the count only; the process segfaults. Kokkos
// hip.graph_instantiate_and_debug_dot_print sizes its node query this way.
// hipGraphGetRootNodes also reports the total node count instead of the
// number of root nodes, and hipGraphGetNodes fills only the first slot.
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void incrementKernel(int *Data) { ++*Data; }

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  int *Dev;
  CHECK(hipMalloc(&Dev, sizeof(int)));

  void *Args[] = {&Dev};
  hipKernelNodeParams Params = {};
  Params.func = (void *)incrementKernel;
  Params.gridDim = dim3(1);
  Params.blockDim = dim3(1);
  Params.sharedMemBytes = 0;
  Params.kernelParams = Args;
  Params.extra = nullptr;

  // root (empty) -> kernel
  hipGraph_t Graph;
  CHECK(hipGraphCreate(&Graph, 0));
  hipGraphNode_t Root, Kernel;
  CHECK(hipGraphAddEmptyNode(&Root, Graph, nullptr, 0));
  CHECK(hipGraphAddKernelNode(&Kernel, Graph, &Root, 1, &Params));

  size_t NumNodes = 0;
  CHECK(hipGraphGetNodes(Graph, nullptr, &NumNodes));
  if (NumNodes != 2) {
    std::cout << "FAIL: hipGraphGetNodes counted " << NumNodes
              << " nodes, expected 2\n";
    return 2;
  }

  hipGraphNode_t Nodes[2] = {};
  NumNodes = 2;
  CHECK(hipGraphGetNodes(Graph, Nodes, &NumNodes));
  if (NumNodes != 2 || !Nodes[0] || !Nodes[1] || Nodes[0] == Nodes[1]) {
    std::cout << "FAIL: hipGraphGetNodes returned " << NumNodes << " nodes ("
              << Nodes[0] << ", " << Nodes[1] << ")\n";
    return 3;
  }

  size_t NumRoots = 0;
  CHECK(hipGraphGetRootNodes(Graph, nullptr, &NumRoots));
  if (NumRoots != 1) {
    std::cout << "FAIL: hipGraphGetRootNodes counted " << NumRoots
              << " root nodes, expected 1\n";
    return 4;
  }

  hipGraphNode_t Roots[2] = {};
  NumRoots = 2;
  CHECK(hipGraphGetRootNodes(Graph, Roots, &NumRoots));
  if (NumRoots != 1 || Roots[0] != Root) {
    std::cout << "FAIL: hipGraphGetRootNodes returned " << NumRoots
              << " root nodes, first " << Roots[0] << ", expected " << Root
              << "\n";
    return 5;
  }

  CHECK(hipGraphDestroy(Graph));
  CHECK(hipFree(Dev));
  std::cout << "PASSED\n";
  return 0;
}
