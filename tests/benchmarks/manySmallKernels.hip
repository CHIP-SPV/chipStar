// Benchmark: hipLaunchKernel overhead with many streams
//
// Measures per-launch time on the default (null) stream as the number of
// co-existing blocking streams increases.  Each blocking stream is idle
// (never receives a kernel), so the runtime must check every one of them
// for cross-queue dependencies on every null-stream launch.
//
// Pre-PR: addDependenciesQueueSyncImpl calls CreateMarkerInQueue for each
//   non-empty blocking stream AND calls isEmptyQueue for every stream.
//   With many idle streams the marker-creation overhead dominates.
//
// Post-PR: isEmptyQueue() returns true for idle streams so marker creation
//   is skipped.  Per-launch time stays flat regardless of stream count.
//
// Usage: ./manySmallKernels [--csv]
//   --csv  print machine-readable CSV (streams, ns_per_launch)

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(call)                                                     \
  do {                                                                  \
    hipError_t _e = (call);                                             \
    if (_e != hipSuccess) {                                             \
      fprintf(stderr, "HIP error %s at %s:%d\n",                       \
              hipGetErrorString(_e), __FILE__, __LINE__);               \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

__global__ void nopKernel() {}

// Return median of a sorted vector.
static double median(std::vector<double> &v) {
  size_t n = v.size();
  return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

static double measureLaunchNs(int numIdleStreams, int warmup, int iters) {
  // Create idle blocking streams.
  std::vector<hipStream_t> idle(numIdleStreams);
  for (auto &s : idle)
    CHECK(hipStreamCreate(&s));

  // Warm up: flush JIT and runtime state.
  for (int i = 0; i < warmup; ++i)
    nopKernel<<<1, 1, 0, nullptr>>>();
  CHECK(hipDeviceSynchronize());

  // Timed loop: launch on null stream, sync each iteration so the queue
  // is truly empty on the next launch (worst case for cross-queue checks).
  std::vector<double> samples(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    nopKernel<<<1, 1, 0, nullptr>>>();
#ifndef CHECK_ONLY_LAUNCH
    CHECK(hipStreamSynchronize(nullptr));
    auto t1 = std::chrono::steady_clock::now();
#else
    auto t1 = std::chrono::steady_clock::now();
    CHECK(hipStreamSynchronize(nullptr));
#endif
    samples[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
  }

  for (auto &s : idle)
    CHECK(hipStreamDestroy(s));

  std::sort(samples.begin(), samples.end());
  return median(samples);
}

// Maximum ratio of any measurement to the 0-stream baseline before the
// test fails.  The fixed implementation shows ~1x flat; the broken one
// shows 4-30x at 64 streams.  3x gives comfortable margin in both directions.
static const double SCALE_FAIL_THRESHOLD = 3.0;

int main(int argc, char **argv) {
  bool csv = (argc > 1 && strcmp(argv[1], "--csv") == 0);

  // Stream counts to sweep.
  const int counts[] = {0, 1, 2, 4, 8, 16, 32, 64};
  const int N        = sizeof(counts) / sizeof(counts[0]);
  const int WARMUP   = 20;
  const int ITERS    = 200;

  if (!csv) {
    printf("hipLaunchKernel overhead benchmark  (%d samples per point)\n", ITERS);
    printf("Null-stream launch + sync while N idle blocking streams exist\n\n");
    printf("  Idle streams | Median ns/launch\n");
    printf("  -------------|------------------\n");
  } else {
    printf("idle_streams,median_ns_per_launch\n");
  }

  double results[N];
  double base = 0.0;
  for (int i = 0; i < N; ++i) {
    results[i] = measureLaunchNs(counts[i], WARMUP, ITERS);
    if (i == 0) base = results[i];
    if (!csv)
      printf("  %12d | %10.1f ns  (%.2fx baseline)\n",
             counts[i], results[i], results[i] / base);
    else
      printf("%d,%.1f\n", counts[i], results[i]);
  }

  // Fail if any stream count shows more than SCALE_FAIL_THRESHOLD x the
  // 0-stream baseline — indicates O(N) launch overhead regression.
  bool failed = false;
  for (int i = 1; i < N; ++i) {
    if (results[i] > SCALE_FAIL_THRESHOLD * base) {
      if (!csv)
        printf("\nFAIL: %d idle streams = %.1f ns (%.2fx baseline, threshold %.1fx)\n",
               counts[i], results[i], results[i] / base, SCALE_FAIL_THRESHOLD);
      failed = true;
    }
  }

  if (!csv) {
    printf(failed ? "\nFAIL\n" : "\nPASS\n");
  }
  return failed ? 1 : 0;
}
