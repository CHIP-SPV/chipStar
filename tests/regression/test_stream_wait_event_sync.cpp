// Reproducer: hipStreamWaitEvent incorrectly forces stream into
// hipStreamCaptureStatusActive state, causing the next
// hipStreamSynchronize to return hipErrorStreamCaptureInvalidated (901).
//
// Observed in hipMM DEVICE_MR_REF_TEST Binning/AllocateOnStream:
//   pool_memory_resource (stream_ordered_memory_resource) calls
//   cudaStreamWaitEvent on a user stream waiting for an event recorded on
//   a different stream; the subsequent stream.synchronize() throws
//   "hipErrorUnknown" (actually 901 = hipErrorStreamCaptureInvalidated).
//
// Root cause: CHIPBindings.cc:3758-3759 in hipStreamWaitEventInternal
// unconditionally flips the target stream's capture status to
// hipStreamCaptureStatusActive, even when no graph capture was ever
// started on the stream. The subsequent hipStreamSynchronize then sees
// getCaptureStatus() != None and returns hipErrorStreamCaptureInvalidated.
//
// Expected: both hipStreamWaitEvent and hipStreamSynchronize succeed.
#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(expr) do {                                                    \
  hipError_t e = (expr);                                                    \
  if (e != hipSuccess) {                                                    \
    printf("FAIL %s = %d (%s)\n", #expr, (int)e, hipGetErrorString(e));     \
    return 1;                                                               \
  }                                                                         \
} while (0)

int main() {
  hipStream_t s;
  CHECK(hipStreamCreate(&s));

  hipEvent_t ev;
  CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));

  // Record the event on the default (legacy/null) stream.
  CHECK(hipEventRecord(ev, 0));

  // Make the user stream wait for the default-stream event.
  // This path is used heavily by RMM-style pool allocators.
  CHECK(hipStreamWaitEvent(s, ev, 0));

  // Without the hipStreamWaitEvent bug, this succeeds. With it, the
  // implementation has put `s` into hipStreamCaptureStatusActive, so
  // this returns 901 (hipErrorStreamCaptureInvalidated).
  CHECK(hipStreamSynchronize(s));

  CHECK(hipEventDestroy(ev));
  CHECK(hipStreamDestroy(s));
  printf("PASS\n");
  return 0;
}
