// Reproduces: hipGraphLaunch fails with hipErrorTbd ("Failed to find" from
// CHIPGraphNode::removeDependency) on a graph whose redundant edges are
// reachable through more than one path. Surfaced by Kokkos
// hip_graph.when_all_cycle, which wires exactly this shape:
//
//   root -> f1 -> f2 -> f3
//   agg1 depends on {f2, f3};  f4 depends on agg1
//   agg2 depends on {f1, f4, f3};  f5 depends on agg2
//
// The edges agg2->f1, agg2->f3 and agg1->f2 are all implied by longer paths.
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void setTo(int *Dst, int Val) { *Dst = Val; }
__global__ void addOne(int *Dst) { atomicAdd(Dst, 1); }

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static hipKernelNodeParams kernelParams(const void *Func, void **Args,
                                        unsigned Grid) {
  hipKernelNodeParams Params = {};
  Params.func = const_cast<void *>(Func);
  Params.gridDim = dim3(Grid);
  Params.blockDim = dim3(1);
  Params.sharedMemBytes = 0;
  Params.kernelParams = Args;
  Params.extra = nullptr;
  return Params;
}

int main() {
  int *Count;
  CHECK(hipMalloc(&Count, sizeof(int)));
  CHECK(hipMemset(Count, 0xff, sizeof(int)));
  CHECK(hipDeviceSynchronize());

  hipGraph_t Graph;
  CHECK(hipGraphCreate(&Graph, 0));

  int Zero = 0;
  void *SetArgs[] = {&Count, &Zero};
  void *AddArgs[] = {&Count};
  hipKernelNodeParams SetParams = kernelParams((const void *)setTo, SetArgs, 1);
  hipKernelNodeParams AddParams = kernelParams((const void *)addOne, AddArgs, 1);
  hipKernelNodeParams Add5Params =
      kernelParams((const void *)addOne, AddArgs, 5);

  // Kokkos adds every node without dependencies and wires the edges afterwards
  // with hipGraphAddDependencies, one edge per call.
  hipGraphNode_t Root, F1, F2, F3, Agg1, F4, Agg2, F5;
  CHECK(hipGraphAddEmptyNode(&Root, Graph, nullptr, 0));
  CHECK(hipGraphAddKernelNode(&F1, Graph, nullptr, 0, &SetParams));
  CHECK(hipGraphAddKernelNode(&F2, Graph, nullptr, 0, &AddParams));
  CHECK(hipGraphAddKernelNode(&F3, Graph, nullptr, 0, &Add5Params));
  CHECK(hipGraphAddEmptyNode(&Agg1, Graph, nullptr, 0));
  CHECK(hipGraphAddKernelNode(&F4, Graph, nullptr, 0, &AddParams));
  CHECK(hipGraphAddEmptyNode(&Agg2, Graph, nullptr, 0));
  CHECK(hipGraphAddKernelNode(&F5, Graph, nullptr, 0, &AddParams));

  const hipGraphNode_t Edges[][2] = {
      {Root, F1}, {F1, F2},   {F2, F3},   {F2, Agg1}, {F3, Agg1}, {Agg1, F4},
      {F1, Agg2}, {F4, Agg2}, {F3, Agg2}, {Agg2, F5},
  };
  for (const auto &Edge : Edges)
    CHECK(hipGraphAddDependencies(Graph, &Edge[0], &Edge[1], 1));

  hipGraphExec_t GraphExec;
  CHECK(hipGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(GraphExec, 0));
  CHECK(hipDeviceSynchronize());

  // set 0, +1 (f2), +5 (f3), +1 (f4), +1 (f5)
  int Host = -1;
  CHECK(hipMemcpy(&Host, Count, sizeof(int), hipMemcpyDeviceToHost));
  if (Host != 8) {
    std::cout << "FAIL: count is " << Host << ", expected 8\n";
    return 2;
  }

  CHECK(hipGraphExecDestroy(GraphExec));
  CHECK(hipGraphDestroy(Graph));
  CHECK(hipFree(Count));
  std::cout << "PASSED\n";
  return 0;
}
