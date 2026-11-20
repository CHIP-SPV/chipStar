// Reproduces: hipGraphLaunch schedules the graph the executable graph was
// instantiated from, so nodes and edges added to that graph after
// hipGraphInstantiate change what the executable graph does.
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

static hipMemsetParams memsetParams(void *Dst, unsigned char Value) {
  hipMemsetParams Params = {};
  Params.dst = Dst;
  Params.value = Value;
  Params.elementSize = 1;
  Params.width = 1;
  Params.height = 1;
  return Params;
}

int main() {
  unsigned char *Dev;
  CHECK(hipMalloc(&Dev, 2));
  CHECK(hipMemset(Dev, 0, 2));

  // Dev[0] = 1, then Dev[0] = 2.
  hipGraph_t Graph;
  hipGraphNode_t First, Second, Added;
  CHECK(hipGraphCreate(&Graph, 0));
  hipMemsetParams P1 = memsetParams(Dev, 1), P2 = memsetParams(Dev, 2);
  CHECK(hipGraphAddMemsetNode(&First, Graph, nullptr, 0, &P1));
  CHECK(hipGraphAddMemsetNode(&Second, Graph, &First, 1, &P2));

  hipGraphExec_t Exec;
  CHECK(hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0));

  // Reverse the edge and add a node that sets Dev[1]; neither may reach Exec.
  CHECK(hipGraphRemoveDependencies(Graph, &First, &Second, 1));
  CHECK(hipGraphAddDependencies(Graph, &Second, &First, 1));
  hipMemsetParams P3 = memsetParams(Dev + 1, 3);
  CHECK(hipGraphAddMemsetNode(&Added, Graph, nullptr, 0, &P3));

  CHECK(hipGraphLaunch(Exec, 0));
  CHECK(hipStreamSynchronize(0));
  unsigned char Host[2];
  CHECK(hipMemcpy(Host, Dev, 2, hipMemcpyDeviceToHost));

  int Failed = 0;
  if (Host[0] != 2) {
    std::cout << "FAIL: Dev[0] is " << int(Host[0])
              << ", expected 2: an edge changed after instantiation was used\n";
    Failed = 1;
  }
  if (Host[1] != 0) {
    std::cout << "FAIL: Dev[1] is " << int(Host[1])
              << ", expected 0: a node added after instantiation ran\n";
    Failed = 1;
  }

  CHECK(hipGraphExecDestroy(Exec));
  CHECK(hipGraphDestroy(Graph));
  CHECK(hipFree(Dev));
  std::cout << (Failed ? "FAILED\n" : "PASSED\n");
  return Failed;
}
