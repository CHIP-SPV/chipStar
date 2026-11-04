// Both memset node setters must reject an elementSize other than 1, 2 or 4.
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
  void *Dev;
  if (hipMalloc(&Dev, 64) != hipSuccess)
    return 1;
  hipMemsetParams P = {};
  P.dst = Dev;
  P.elementSize = 1;
  P.width = 16;
  P.height = 1;
  hipGraph_t Graph;
  hipGraphNode_t Node;
  hipGraphExec_t Exec;
  if (hipGraphCreate(&Graph, 0) != hipSuccess ||
      hipGraphAddMemsetNode(&Node, Graph, nullptr, 0, &P) != hipSuccess ||
      hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0) != hipSuccess) {
    std::cout << "FAIL: graph setup\n";
    return 1;
  }

  int Failures = 0;
  for (unsigned Size : {0u, 1u, 2u, 3u, 4u, 5u, 8u}) {
    P.elementSize = Size;
    hipError_t Want = (Size == 1 || Size == 2 || Size == 4)
                          ? hipSuccess
                          : hipErrorInvalidValue;
    hipError_t Node_ = hipGraphMemsetNodeSetParams(Node, &P);
    hipError_t Exec_ = hipGraphExecMemsetNodeSetParams(Exec, Node, &P);
    if (Node_ != Want || Exec_ != Want) {
      std::cout << "FAIL: elementSize " << Size << " node setter "
                << hipGetErrorName(Node_) << ", exec setter "
                << hipGetErrorName(Exec_) << ", expected "
                << hipGetErrorName(Want) << "\n";
      Failures++;
    }
  }
  if (Failures)
    return 2;
  std::cout << "PASSED\n";
  return 0;
}
