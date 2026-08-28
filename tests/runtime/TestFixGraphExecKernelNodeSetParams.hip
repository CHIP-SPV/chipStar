// Reproduces: hipGraphExecKernelNodeSetParams and hipGraphExecMemsetNodeSetParams
// look the node up in the clone map of the ORIGINAL graph, which is only
// populated when that graph itself came from hipGraphClone, so on a normally
// built graph they fail; and where they do resolve a node they update the
// original graph's node instead of its copy in the executable graph.
// HIP semantics: the update applies to the instantiated graph only, the
// original graph is untouched, and the next hipGraphLaunch of that executable
// graph runs with the new parameters.
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void writeKernel(int *Data, int Value) { *Data = Value; }

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int readBack(int *Dev) {
  int Host = -1;
  (void)hipMemcpy(&Host, Dev, sizeof(int), hipMemcpyDeviceToHost);
  return Host;
}

int main() {
  int *Dev;
  CHECK(hipMalloc(&Dev, sizeof(int)));

  // Kernel node: the second argument is the value the kernel writes.
  int ValueA = 11, ValueB = 22;
  void *ArgsA[] = {&Dev, &ValueA};
  hipKernelNodeParams Params = {};
  Params.func = (void *)writeKernel;
  Params.gridDim = dim3(1);
  Params.blockDim = dim3(1);
  Params.kernelParams = ArgsA;

  hipGraph_t Graph;
  CHECK(hipGraphCreate(&Graph, 0));
  hipGraphNode_t Kernel;
  CHECK(hipGraphAddKernelNode(&Kernel, Graph, nullptr, 0, &Params));
  hipGraphExec_t Exec;
  CHECK(hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0));

  CHECK(hipMemset(Dev, 0, sizeof(int)));
  CHECK(hipGraphLaunch(Exec, 0));
  CHECK(hipDeviceSynchronize());
  if (int V = readBack(Dev); V != ValueA) {
    std::cout << "FAIL: first launch wrote " << V << ", expected " << ValueA
              << "\n";
    return 2;
  }

  void *ArgsB[] = {&Dev, &ValueB};
  hipKernelNodeParams ParamsB = Params;
  ParamsB.kernelParams = ArgsB;
  CHECK(hipGraphExecKernelNodeSetParams(Exec, Kernel, &ParamsB));

  CHECK(hipMemset(Dev, 0, sizeof(int)));
  CHECK(hipGraphLaunch(Exec, 0));
  CHECK(hipDeviceSynchronize());
  if (int V = readBack(Dev); V != ValueB) {
    std::cout << "FAIL: launch after hipGraphExecKernelNodeSetParams wrote "
              << V << ", expected " << ValueB << "\n";
    return 3;
  }

  // The original graph is untouched: its node still carries ValueA and a
  // fresh instantiation still writes it.
  hipKernelNodeParams Original = {};
  CHECK(hipGraphKernelNodeGetParams(Kernel, &Original));
  if (int V = *static_cast<int *>(Original.kernelParams[1]); V != ValueA) {
    std::cout << "FAIL: original kernel node argument is " << V
              << ", expected " << ValueA << "\n";
    return 4;
  }
  hipGraphExec_t Exec2;
  CHECK(hipGraphInstantiate(&Exec2, Graph, nullptr, nullptr, 0));
  CHECK(hipMemset(Dev, 0, sizeof(int)));
  CHECK(hipGraphLaunch(Exec2, 0));
  CHECK(hipDeviceSynchronize());
  if (int V = readBack(Dev); V != ValueA) {
    std::cout << "FAIL: re-instantiated original graph wrote " << V
              << ", expected " << ValueA << "\n";
    return 5;
  }

  // Memset node: the same contract for hipGraphExecMemsetNodeSetParams.
  hipMemsetParams Memset = {};
  Memset.dst = Dev;
  Memset.elementSize = 1;
  Memset.width = sizeof(int);
  Memset.height = 1;
  Memset.value = 0x05;
  hipGraph_t MemsetGraph;
  CHECK(hipGraphCreate(&MemsetGraph, 0));
  hipGraphNode_t MemsetNode;
  CHECK(hipGraphAddMemsetNode(&MemsetNode, MemsetGraph, nullptr, 0, &Memset));
  hipGraphExec_t MemsetExec;
  CHECK(hipGraphInstantiate(&MemsetExec, MemsetGraph, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(MemsetExec, 0));
  CHECK(hipDeviceSynchronize());
  if (int V = readBack(Dev); V != 0x05050505) {
    std::cout << "FAIL: memset launch wrote " << std::hex << V
              << ", expected 0x05050505\n";
    return 6;
  }

  Memset.value = 0x06;
  CHECK(hipGraphExecMemsetNodeSetParams(MemsetExec, MemsetNode, &Memset));
  CHECK(hipGraphLaunch(MemsetExec, 0));
  CHECK(hipDeviceSynchronize());
  if (int V = readBack(Dev); V != 0x06060606) {
    std::cout << "FAIL: launch after hipGraphExecMemsetNodeSetParams wrote "
              << std::hex << V << ", expected 0x06060606\n";
    return 7;
  }

  hipMemsetParams OriginalMemset = {};
  CHECK(hipGraphMemsetNodeGetParams(MemsetNode, &OriginalMemset));
  if (OriginalMemset.value != 0x05) {
    std::cout << "FAIL: original memset node value is " << std::hex
              << OriginalMemset.value << ", expected 0x5\n";
    return 8;
  }

  CHECK(hipGraphExecDestroy(MemsetExec));
  CHECK(hipGraphDestroy(MemsetGraph));
  CHECK(hipGraphExecDestroy(Exec2));
  CHECK(hipGraphExecDestroy(Exec));
  CHECK(hipGraphDestroy(Graph));
  CHECK(hipFree(Dev));
  std::cout << "PASSED\n";
  return 0;
}
