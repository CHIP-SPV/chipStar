// Reproduces: the graph of a stream capture is leaked when the capture is
// abandoned (hipStreamDestroy, or a capture invalidated by hipStreamQuery),
// and an invalidated capture cannot be ended by hipStreamEndCapture.
#include <hip/hip_runtime_api.h>
#include "CHIPBackend.hh"
#include <cstdlib>
#include <iostream>
#include <new>

// Replacement global operator new/delete so the test can tell when the
// capture graph is freed.
static const void *Watched = nullptr;
static bool WatchedFreed = false;
void *operator new(std::size_t Size) {
  if (void *P = std::malloc(Size ? Size : 1))
    return P;
  throw std::bad_alloc();
}
void operator delete(void *P) noexcept {
  if (P && P == Watched)
    WatchedFreed = true;
  std::free(P);
}
void operator delete(void *P, std::size_t) noexcept { operator delete(P); }

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static chipstar::Queue *queue(hipStream_t S) {
  return static_cast<chipstar::Queue *>(S);
}

int main() {
  int Failed = 0;
  hipStream_t S;

  // Destroying a stream in the middle of its capture frees the graph.
  CHECK(hipStreamCreate(&S));
  CHECK(hipStreamBeginCapture(S, hipStreamCaptureModeRelaxed));
  Watched = queue(S)->getCaptureGraph();
  WatchedFreed = false;
  CHECK(hipStreamDestroy(S));
  if (!WatchedFreed) {
    std::cout << "FAIL: hipStreamDestroy leaked the capture graph\n";
    Failed = 1;
  }

  // An invalidated capture cannot be restarted, and ending it frees the graph
  // and leaves the stream out of capture.
  CHECK(hipStreamCreate(&S));
  CHECK(hipStreamBeginCapture(S, hipStreamCaptureModeRelaxed));
  Watched = queue(S)->getCaptureGraph();
  WatchedFreed = false;
  hipError_t Err = hipStreamQuery(S);
  if (Err != hipErrorStreamCaptureInvalidated) {
    std::cout << "FAIL: hipStreamQuery returned " << hipGetErrorString(Err)
              << ", expected hipErrorStreamCaptureInvalidated\n";
    return 1;
  }
  Err = hipStreamBeginCapture(S, hipStreamCaptureModeRelaxed);
  if (Err != hipErrorIllegalState) {
    std::cout << "FAIL: hipStreamBeginCapture on an invalidated capture "
                 "returned "
              << hipGetErrorString(Err) << ", expected hipErrorIllegalState\n";
    return 1;
  }
  hipGraph_t Graph = reinterpret_cast<hipGraph_t>(1);
  Err = hipStreamEndCapture(S, &Graph);
  if (Err != hipErrorStreamCaptureInvalidated || Graph != nullptr) {
    std::cout << "FAIL: hipStreamEndCapture on an invalidated capture "
                 "returned "
              << hipGetErrorString(Err)
              << ", expected hipErrorStreamCaptureInvalidated and no graph\n";
    Failed = 1;
  }
  if (!WatchedFreed) {
    std::cout << "FAIL: the invalidated capture's graph was leaked\n";
    Failed = 1;
  }
  if (queue(S)->getCaptureStatus() != hipStreamCaptureStatusNone) {
    std::cout << "FAIL: the stream is still capturing\n";
    Failed = 1;
  }
  CHECK(hipStreamDestroy(S));

  // A stream whose part of a capture was invalidated cannot rejoin it: the
  // origin would no longer reach the streams forked from it.
  hipStream_t Fork;
  hipEvent_t Event;
  CHECK(hipStreamCreate(&S));
  CHECK(hipStreamCreate(&Fork));
  CHECK(hipEventCreate(&Event));
  CHECK(hipStreamBeginCapture(S, hipStreamCaptureModeRelaxed));
  CHECK(hipEventRecord(Event, S));
  CHECK(hipStreamWaitEvent(Fork, Event, 0));
  hipStreamQuery(Fork);
  Err = hipStreamWaitEvent(Fork, Event, 0);
  if (Err != hipErrorStreamCaptureInvalidated) {
    std::cout << "FAIL: an invalidated stream rejoined the capture: "
              << hipGetErrorString(Err) << "\n";
    Failed = 1;
  }
  hipStreamEndCapture(S, &Graph);
  CHECK(hipEventDestroy(Event));
  CHECK(hipStreamDestroy(Fork));
  CHECK(hipStreamDestroy(S));

  std::cout << (Failed ? "FAILED\n" : "PASSED\n");
  return Failed;
}
