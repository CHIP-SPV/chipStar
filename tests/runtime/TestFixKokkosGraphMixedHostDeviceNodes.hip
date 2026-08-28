// Reproduces: hipLaunchHostFunc on a capturing stream runs the host function
// right away instead of recording a host node, so a captured chain of kernel
// launches and host functions runs its host steps during capture and the
// graph replays only the kernels.
// The motivating case is Kokkos hip.mixed_then_host_device_nodes, which
// chains then (kernel) -> then_host -> then on a hipMallocManaged counter,
// each node adding 2, and reads 4 instead of 6 on Aurora. Kokkos then_host
// uses hipGraphAddHostNode, so that chain is checked here as well, built
// the way Kokkos builds it.
#include <hip/hip_runtime.h>
#include <iostream>

// Each step appends its own digit to the counter, so the final value spells
// out the order in which the nodes actually ran: K H K H K H reads 121212.
__global__ void deviceStep(unsigned *Counter) { *Counter = *Counter * 10 + 1; }

static void hostStep(void *Data) {
  unsigned *Counter = static_cast<unsigned *>(Data);
  *Counter = *Counter * 10 + 2;
}

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static constexpr unsigned Expected = 121212;

int main() {
  unsigned *Counter;
  CHECK(hipMallocManaged(&Counter, sizeof(unsigned)));
  hipStream_t Stream;
  CHECK(hipStreamCreate(&Stream));

  void *Args[] = {&Counter};
  hipKernelNodeParams KernelParams = {};
  KernelParams.func = (void *)deviceStep;
  KernelParams.gridDim = dim3(1);
  KernelParams.blockDim = dim3(1);
  KernelParams.kernelParams = Args;
  hipHostNodeParams HostParams = {};
  HostParams.fn = hostStep;
  HostParams.userData = Counter;

  // root (empty) -> K -> H -> K -> H -> K -> H, every node added without
  // dependencies and chained afterwards through hipGraphAddDependencies, the
  // way Kokkos builds its graphs.
  *Counter = 0;
  hipGraph_t Graph;
  CHECK(hipGraphCreate(&Graph, 0));
  hipGraphNode_t Prev;
  CHECK(hipGraphAddEmptyNode(&Prev, Graph, nullptr, 0));
  for (int I = 0; I < 3; ++I) {
    hipGraphNode_t Kernel, Host;
    CHECK(hipGraphAddKernelNode(&Kernel, Graph, nullptr, 0, &KernelParams));
    CHECK(hipGraphAddDependencies(Graph, &Prev, &Kernel, 1));
    CHECK(hipGraphAddHostNode(&Host, Graph, nullptr, 0, &HostParams));
    CHECK(hipGraphAddDependencies(Graph, &Kernel, &Host, 1));
    Prev = Host;
  }
  hipGraphExec_t GraphExec;
  CHECK(hipGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(GraphExec, Stream));
  CHECK(hipStreamSynchronize(Stream));
  if (*Counter != Expected) {
    std::cout << "FAIL: explicit graph ran the nodes as " << *Counter
              << ", expected " << Expected << "\n";
    return 2;
  }
  CHECK(hipGraphExecDestroy(GraphExec));
  CHECK(hipGraphDestroy(Graph));

  // The same chain recorded by stream capture.
  *Counter = 0;
  CHECK(hipStreamBeginCapture(Stream, hipStreamCaptureModeGlobal));
  for (int I = 0; I < 3; ++I) {
    deviceStep<<<1, 1, 0, Stream>>>(Counter);
    CHECK(hipLaunchHostFunc(Stream, hostStep, Counter));
  }
  CHECK(hipStreamEndCapture(Stream, &Graph));
  CHECK(hipStreamSynchronize(Stream));
  if (*Counter != 0) {
    std::cout << "FAIL: captured work ran during capture, counter is "
              << *Counter << "\n";
    return 3;
  }
  CHECK(hipGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(GraphExec, Stream));
  CHECK(hipStreamSynchronize(Stream));
  if (*Counter != Expected) {
    std::cout << "FAIL: captured graph ran the nodes as " << *Counter
              << ", expected " << Expected << "\n";
    return 4;
  }
  CHECK(hipGraphExecDestroy(GraphExec));
  CHECK(hipGraphDestroy(Graph));

  CHECK(hipStreamDestroy(Stream));
  CHECK(hipFree(Counter));
  std::cout << "PASSED\n";
  return 0;
}
