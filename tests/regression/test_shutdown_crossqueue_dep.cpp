// Reproducer for issue #1311: heap corruption ("malloc(): unsorted double
// linked list corrupted", SIGABRT) at process shutdown -- a regression from
// the shared-immediate-command-list change (#1293).
//
// Mechanism: mixing the default stream with explicit blocking streams makes
// CHIPQueueLevel0 store cross-queue sync marker events in
// PendingCrossQueueDeps_. Those markers' shared_ptr deleter is
// LZEventPool::returnEvent, which pushes the raw event back onto the owning
// pool. CHIPContextLevel0::~CHIPContextLevel0 must delete the device (and thus
// its queues, releasing those markers back into the pools) BEFORE deleting the
// event pools. The pre-fix ordering freed the pools first, so returnEvent
// pushed onto freed memory at teardown and corrupted the heap.
//
// In normal runs chipStar's shutdown safety net (CHIPUninitializeCallOnce ->
// Queue::finish() -> PendingCrossQueueDeps_.clear()) hides the bug. Zero-RK
// defeats that net (a finish() whose Level Zero sync throws before the clear).
// This test reproduces that condition deterministically via the test-only env
// var CHIP_L0_TEST_RETAIN_CROSSQUEUE_DEPS, which makes finish() retain the
// markers. It then exits WITHOUT destroying streams/events so all teardown runs
// through the runtime shutdown destructors -- the path that corrupted the heap.
//
// Expected: exits cleanly ("PASS"). Pre-fix: aborts in libc during shutdown.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(expr) do {                                                    \
  hipError_t e = (expr);                                                    \
  if (e != hipSuccess) {                                                    \
    printf("FAIL %s = %d (%s)\n", #expr, (int)e, hipGetErrorString(e));     \
    return 1;                                                               \
  }                                                                         \
} while (0)

__global__ void scale(float *a) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  a[i] = a[i] * 2.0f + 1.0f;
}

int main() {
  // Force the cross-queue dependency markers to survive until teardown,
  // reproducing the Zero-RK shutdown condition. Must be set before any HIP
  // call so the runtime observes it on the first finish().
  setenv("CHIP_L0_TEST_RETAIN_CROSSQUEUE_DEPS", "1", 1);

  const int NS = 16;
  const int N = 1 << 14;
  const size_t sz = N * sizeof(float);

  float *d;
  CHECK(hipMalloc(&d, sz));
  std::vector<float *> bufs(NS);
  std::vector<hipStream_t> streams(NS);
  for (int i = 0; i < NS; i++) {
    CHECK(hipMalloc(&bufs[i], sz));
    CHECK(hipStreamCreate(&streams[i]));
  }

  // Interleave default-stream and blocking-stream work so each queue
  // accumulates cross-queue dependency markers.
  for (int it = 0; it < 64; ++it) {
    hipLaunchKernelGGL(scale, dim3(N / 256), dim3(256), 0, 0, d);
    for (int i = 0; i < NS; i++)
      hipLaunchKernelGGL(scale, dim3(N / 256), dim3(256), 0, streams[i], bufs[i]);
  }

  CHECK(hipDeviceSynchronize());

  // Intentionally no hipStreamDestroy / hipFree: leave teardown to the runtime
  // shutdown destructors, which is where #1311 corrupted the heap.
  printf("PASS\n");
  return 0;
}
