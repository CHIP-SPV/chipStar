// Reproduces: a kernel launched into a stream that is under
// hipStreamBeginCapture runs immediately instead of being recorded into the
// capture graph, and hipStreamEndCapture hands back an empty graph.
// Surfaced by Kokkos hip.graph_capture ("0 is not in data used in the
// captured kernel, got 5"): the captured kernel had already written its value
// before the graph was ever launched.
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void setTo(int *Dst, int Val) { *Dst = Val; }

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  hipStream_t Stream;
  CHECK(hipStreamCreate(&Stream));

  int *Dev;
  CHECK(hipMalloc(&Dev, sizeof(int)));
  CHECK(hipMemset(Dev, 0, sizeof(int)));
  CHECK(hipDeviceSynchronize());

  hipGraph_t Graph;
  CHECK(hipStreamBeginCapture(Stream, hipStreamCaptureModeGlobal));
  setTo<<<1, 1, 0, Stream>>>(Dev, 5);
  CHECK(hipGetLastError());
  CHECK(hipStreamEndCapture(Stream, &Graph));
  CHECK(hipStreamSynchronize(Stream));

  // Nothing has been launched yet: the value must still be 0.
  int Host = -1;
  CHECK(hipMemcpy(&Host, Dev, sizeof(int), hipMemcpyDeviceToHost));
  if (Host != 0) {
    std::cout << "FAIL: kernel ran during capture, value is " << Host << "\n";
    return 2;
  }

  // The launch must have been recorded as one kernel node.
  hipGraphNode_t Nodes[2] = {};
  size_t NumNodes = 2;
  CHECK(hipGraphGetNodes(Graph, Nodes, &NumNodes));
  if (NumNodes != 1) {
    std::cout << "FAIL: captured graph has " << NumNodes
              << " nodes, expected 1\n";
    return 3;
  }

  // Replaying the captured graph performs the write.
  hipGraphExec_t GraphExec;
  CHECK(hipGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(GraphExec, Stream));
  CHECK(hipStreamSynchronize(Stream));
  CHECK(hipMemcpy(&Host, Dev, sizeof(int), hipMemcpyDeviceToHost));
  if (Host != 5) {
    std::cout << "FAIL: graph launch did not run the captured kernel, value is "
              << Host << "\n";
    return 4;
  }

  // A second capture on the same stream starts a fresh graph: its node must
  // not be chained behind the node recorded by the previous capture.
  hipGraph_t Graph2;
  CHECK(hipStreamBeginCapture(Stream, hipStreamCaptureModeGlobal));
  setTo<<<1, 1, 0, Stream>>>(Dev, 7);
  CHECK(hipGetLastError());
  CHECK(hipStreamEndCapture(Stream, &Graph2));
  NumNodes = 2;
  CHECK(hipGraphGetNodes(Graph2, Nodes, &NumNodes));
  if (NumNodes != 1) {
    std::cout << "FAIL: second captured graph has " << NumNodes
              << " nodes, expected 1\n";
    return 5;
  }
  size_t NumDeps = 1;
  CHECK(hipGraphNodeGetDependencies(Nodes[0], nullptr, &NumDeps));
  if (NumDeps != 0) {
    std::cout << "FAIL: node of the second capture has " << NumDeps
              << " dependencies, expected 0\n";
    return 6;
  }

  CHECK(hipGraphDestroy(Graph2));
  CHECK(hipGraphExecDestroy(GraphExec));
  CHECK(hipGraphDestroy(Graph));
  CHECK(hipFree(Dev));
  CHECK(hipStreamDestroy(Stream));
  std::cout << "PASSED\n";
  return 0;
}
