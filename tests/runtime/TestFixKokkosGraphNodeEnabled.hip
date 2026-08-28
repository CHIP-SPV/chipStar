// Reproduces: hipGraphNodeSetEnabled and hipGraphNodeGetEnabled return
// hipErrorNotSupported. Kokkos hip.interact_with_hip_node disables the kernel
// node of an instantiated graph, submits, expects the kernel not to have run,
// re-enables it, submits again and expects it to have run.
// The flag belongs to the executable graph: only kernel, memcpy and memset
// nodes accept it, and a disabled node is a no-op at hipGraphLaunch.
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

static int readBack(int *Dev) {
  int Host = -1;
  (void)hipMemcpy(&Host, Dev, sizeof(int), hipMemcpyDeviceToHost);
  return Host;
}

int main() {
  int *Dev;
  CHECK(hipMalloc(&Dev, sizeof(int)));
  CHECK(hipMemset(Dev, 0, sizeof(int)));
  CHECK(hipDeviceSynchronize());

  void *Args[] = {&Dev};
  hipKernelNodeParams Params = {};
  Params.func = (void *)incrementKernel;
  Params.gridDim = dim3(1);
  Params.blockDim = dim3(1);
  Params.sharedMemBytes = 0;
  Params.kernelParams = Args;
  Params.extra = nullptr;

  // root (empty) -> kernel, the shape Kokkos builds for one then_parallel_for
  hipGraph_t Graph;
  CHECK(hipGraphCreate(&Graph, 0));
  hipGraphNode_t Root, Kernel;
  CHECK(hipGraphAddEmptyNode(&Root, Graph, nullptr, 0));
  CHECK(hipGraphAddKernelNode(&Kernel, Graph, &Root, 1, &Params));

  hipGraphExec_t GraphExec;
  CHECK(hipGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0));

  unsigned Enabled = 0;
  CHECK(hipGraphNodeGetEnabled(GraphExec, Kernel, &Enabled));
  if (Enabled != 1) {
    std::cout << "FAIL: fresh kernel node reports enabled=" << Enabled << "\n";
    return 2;
  }

  CHECK(hipGraphLaunch(GraphExec, 0));
  CHECK(hipDeviceSynchronize());
  if (int V = readBack(Dev); V != 1) {
    std::cout << "FAIL: enabled node did not run, value is " << V << "\n";
    return 3;
  }

  CHECK(hipGraphNodeSetEnabled(GraphExec, Kernel, 0));
  CHECK(hipGraphNodeGetEnabled(GraphExec, Kernel, &Enabled));
  if (Enabled != 0) {
    std::cout << "FAIL: disabled kernel node reports enabled=" << Enabled
              << "\n";
    return 4;
  }

  CHECK(hipGraphLaunch(GraphExec, 0));
  CHECK(hipDeviceSynchronize());
  if (int V = readBack(Dev); V != 1) {
    std::cout << "FAIL: disabled node ran, value is " << V << "\n";
    return 5;
  }

  CHECK(hipGraphNodeSetEnabled(GraphExec, Kernel, 1));
  CHECK(hipGraphNodeGetEnabled(GraphExec, Kernel, &Enabled));
  if (Enabled != 1) {
    std::cout << "FAIL: re-enabled kernel node reports enabled=" << Enabled
              << "\n";
    return 6;
  }

  CHECK(hipGraphLaunch(GraphExec, 0));
  CHECK(hipDeviceSynchronize());
  if (int V = readBack(Dev); V != 2) {
    std::cout << "FAIL: re-enabled node did not run, value is " << V << "\n";
    return 7;
  }

  // The flag is only defined for kernel, memcpy and memset nodes.
  if (hipError_t Err = hipGraphNodeSetEnabled(GraphExec, Root, 0);
      Err != hipErrorInvalidValue) {
    std::cout << "FAIL: hipGraphNodeSetEnabled on an empty node -> "
              << hipGetErrorString(Err) << ", expected hipErrorInvalidValue\n";
    return 8;
  }

  CHECK(hipGraphExecDestroy(GraphExec));
  CHECK(hipGraphDestroy(Graph));
  CHECK(hipFree(Dev));
  std::cout << "PASSED\n";
  return 0;
}
