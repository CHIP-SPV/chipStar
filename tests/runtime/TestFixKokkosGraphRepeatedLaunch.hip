// Reproduces: the N-th hipGraphLaunch of the same hipGraphExec_t executes
// the graph N times. Every launch recompiles the execution schedule and
// appends it to the schedule of the previous launches instead of replacing
// it. Kokkos hip_graph.repeat_chain submits one graph ten times.
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void addOne(int *Dst) { atomicAdd(Dst, 1); }

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  int *Count;
  CHECK(hipMalloc(&Count, sizeof(int)));
  CHECK(hipMemset(Count, 0, sizeof(int)));
  CHECK(hipDeviceSynchronize());

  void *Args[] = {&Count};
  hipKernelNodeParams Params = {};
  Params.func = (void *)addOne;
  Params.gridDim = dim3(1);
  Params.blockDim = dim3(1);
  Params.sharedMemBytes = 0;
  Params.kernelParams = Args;
  Params.extra = nullptr;

  hipGraph_t Graph;
  CHECK(hipGraphCreate(&Graph, 0));
  hipGraphNode_t Node;
  CHECK(hipGraphAddKernelNode(&Node, Graph, nullptr, 0, &Params));

  hipGraphExec_t GraphExec;
  CHECK(hipGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0));

  const int Launches = 3;
  for (int i = 0; i < Launches; i++)
    CHECK(hipGraphLaunch(GraphExec, 0));
  CHECK(hipDeviceSynchronize());

  int Host = -1;
  CHECK(hipMemcpy(&Host, Count, sizeof(int), hipMemcpyDeviceToHost));
  if (Host != Launches) {
    std::cout << "FAIL: kernel node ran " << Host << " times over "
              << Launches << " launches\n";
    return 2;
  }

  CHECK(hipGraphExecDestroy(GraphExec));
  CHECK(hipGraphDestroy(Graph));
  CHECK(hipFree(Count));
  std::cout << "PASSED\n";
  return 0;
}
