// Reproduces: work forked onto another stream during stream capture is not
// captured. With capture active on Stream0, hipEventRecord(Ev, Stream0) plus
// hipStreamWaitEvent(Stream1, Ev) must pull Stream1 into the same capture so
// that kernels launched on Stream1 become graph nodes, and joining back with
// another event must make the origin stream depend on them. Surfaced by the
// catch tests Unit_hipStreamBeginCapture_nestedStreamCapture and
// Unit_hipStreamBeginCapture_streamReuse (2 == 7): only the kernels launched
// on the origin stream were recorded.
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void setSlot(int *Dst, int Slot) { Dst[Slot] = 1; }

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  hipStream_t Streams[3];
  hipEvent_t Fork, JoinA, JoinB;
  for (auto &S : Streams)
    CHECK(hipStreamCreate(&S));
  CHECK(hipEventCreate(&Fork));
  CHECK(hipEventCreate(&JoinA));
  CHECK(hipEventCreate(&JoinB));

  int *Dev;
  CHECK(hipMalloc(&Dev, 3 * sizeof(int)));
  CHECK(hipMemset(Dev, 0, 3 * sizeof(int)));
  CHECK(hipDeviceSynchronize());

  // Fork Streams[1] and Streams[2] off the capturing Streams[0], launch one
  // kernel on each of the three streams, then join both forks back.
  hipGraph_t Graph;
  CHECK(hipStreamBeginCapture(Streams[0], hipStreamCaptureModeGlobal));
  CHECK(hipEventRecord(Fork, Streams[0]));
  CHECK(hipStreamWaitEvent(Streams[1], Fork, 0));
  CHECK(hipStreamWaitEvent(Streams[2], Fork, 0));
  setSlot<<<1, 1, 0, Streams[0]>>>(Dev, 0);
  CHECK(hipGetLastError());
  setSlot<<<1, 1, 0, Streams[1]>>>(Dev, 1);
  CHECK(hipGetLastError());
  setSlot<<<1, 1, 0, Streams[2]>>>(Dev, 2);
  CHECK(hipGetLastError());
  CHECK(hipEventRecord(JoinA, Streams[1]));
  CHECK(hipEventRecord(JoinB, Streams[2]));
  CHECK(hipStreamWaitEvent(Streams[0], JoinA, 0));
  CHECK(hipStreamWaitEvent(Streams[0], JoinB, 0));
  CHECK(hipStreamEndCapture(Streams[0], &Graph));
  CHECK(hipDeviceSynchronize());

  // Nothing may have run yet, on any of the three streams.
  int Host[3] = {-1, -1, -1};
  CHECK(hipMemcpy(Host, Dev, sizeof(Host), hipMemcpyDeviceToHost));
  if (Host[0] != 0 || Host[1] != 0 || Host[2] != 0) {
    std::cout << "FAIL: work ran during capture, slots are " << Host[0] << " "
              << Host[1] << " " << Host[2] << "\n";
    return 2;
  }

  // All three kernel launches must have been recorded as nodes.
  size_t NumNodes = 0;
  CHECK(hipGraphGetNodes(Graph, nullptr, &NumNodes));
  if (NumNodes != 3) {
    std::cout << "FAIL: captured graph has " << NumNodes
              << " nodes, expected 3\n";
    return 3;
  }

  // Replaying the graph performs all three writes.
  hipGraphExec_t GraphExec;
  CHECK(hipGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(GraphExec, Streams[0]));
  CHECK(hipStreamSynchronize(Streams[0]));
  CHECK(hipMemcpy(Host, Dev, sizeof(Host), hipMemcpyDeviceToHost));
  if (Host[0] != 1 || Host[1] != 1 || Host[2] != 1) {
    std::cout << "FAIL: graph launch did not run the forked kernels, slots are "
              << Host[0] << " " << Host[1] << " " << Host[2] << "\n";
    return 4;
  }
  CHECK(hipGraphExecDestroy(GraphExec));
  CHECK(hipGraphDestroy(Graph));

  // A fork that is never joined back must be rejected at the end of capture.
  hipGraph_t Unjoined = nullptr;
  CHECK(hipStreamBeginCapture(Streams[0], hipStreamCaptureModeGlobal));
  setSlot<<<1, 1, 0, Streams[0]>>>(Dev, 0);
  CHECK(hipGetLastError());
  CHECK(hipEventRecord(Fork, Streams[0]));
  CHECK(hipStreamWaitEvent(Streams[1], Fork, 0));
  setSlot<<<1, 1, 0, Streams[1]>>>(Dev, 1);
  CHECK(hipGetLastError());
  hipError_t Err = hipStreamEndCapture(Streams[0], &Unjoined);
  if (Err != hipErrorStreamCaptureUnjoined) {
    std::cout << "FAIL: hipStreamEndCapture with an unjoined fork returned "
              << hipGetErrorString(Err) << ", expected "
              << hipGetErrorString(hipErrorStreamCaptureUnjoined) << "\n";
    return 5;
  }

  CHECK(hipFree(Dev));
  CHECK(hipEventDestroy(Fork));
  CHECK(hipEventDestroy(JoinA));
  CHECK(hipEventDestroy(JoinB));
  for (auto &S : Streams)
    CHECK(hipStreamDestroy(S));
  std::cout << "PASSED\n";
  return 0;
}
