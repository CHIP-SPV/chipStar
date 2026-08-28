// Reproduces: hipGraphLaunch on a graph holding a child graph node
// (hipGraphAddChildGraphNode) fails with hipErrorTbd ("Attemped to execute
// GraphNode") instead of running the child's nodes in place. Kokkos wraps
// every stream-captured region in a child graph node
// (GraphNodeCaptureImpl::capture), so hip.graph_capture cannot submit.
// The child node is added last and wired afterwards, as Kokkos does.
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void setTo(int *Dst, int Val) { *Dst = Val; }
__global__ void mulBy(int *Dst, int Val) { *Dst *= Val; }
__global__ void addTo(int *Dst, int Val) { *Dst += Val; }

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static hipKernelNodeParams kernelParams(const void *Func, void **Args) {
  hipKernelNodeParams Params = {};
  Params.func = const_cast<void *>(Func);
  Params.gridDim = dim3(1);
  Params.blockDim = dim3(1);
  Params.sharedMemBytes = 0;
  Params.kernelParams = Args;
  Params.extra = nullptr;
  return Params;
}

int main() {
  int *Dev;
  CHECK(hipMalloc(&Dev, sizeof(int)));
  CHECK(hipMemset(Dev, 0, sizeof(int)));
  CHECK(hipDeviceSynchronize());

  int One = 1, Three = 3, Ten = 10;
  void *SetArgs[] = {&Dev, &One};
  void *MulArgs[] = {&Dev, &Three};
  void *AddArgs[] = {&Dev, &Ten};
  hipKernelNodeParams SetParams = kernelParams((const void *)setTo, SetArgs);
  hipKernelNodeParams MulParams = kernelParams((const void *)mulBy, MulArgs);
  hipKernelNodeParams AddParams = kernelParams((const void *)addTo, AddArgs);

  // Child graph: a single kernel that triples the value.
  hipGraph_t Child;
  CHECK(hipGraphCreate(&Child, 0));
  hipGraphNode_t Mul;
  CHECK(hipGraphAddKernelNode(&Mul, Child, nullptr, 0, &MulParams));

  // Parent graph: set 1, run the child, add 10. Only 1 * 3 + 10 proves the
  // child ran, and ran between its predecessor and its successor.
  hipGraph_t Parent;
  CHECK(hipGraphCreate(&Parent, 0));
  hipGraphNode_t Set, Add, ChildNode;
  CHECK(hipGraphAddKernelNode(&Set, Parent, nullptr, 0, &SetParams));
  CHECK(hipGraphAddKernelNode(&Add, Parent, nullptr, 0, &AddParams));
  CHECK(hipGraphAddChildGraphNode(&ChildNode, Parent, nullptr, 0, Child));
  CHECK(hipGraphAddDependencies(Parent, &Set, &ChildNode, 1));
  CHECK(hipGraphAddDependencies(Parent, &ChildNode, &Add, 1));

  hipGraphExec_t GraphExec;
  CHECK(hipGraphInstantiate(&GraphExec, Parent, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(GraphExec, 0));
  CHECK(hipDeviceSynchronize());

  int Host = -1;
  CHECK(hipMemcpy(&Host, Dev, sizeof(int), hipMemcpyDeviceToHost));
  if (Host != 13) {
    std::cout << "FAIL: value is " << Host << ", expected 13\n";
    return 2;
  }

  CHECK(hipGraphExecDestroy(GraphExec));
  CHECK(hipGraphDestroy(Parent));
  CHECK(hipGraphDestroy(Child));
  CHECK(hipFree(Dev));
  std::cout << "PASSED\n";
  return 0;
}
