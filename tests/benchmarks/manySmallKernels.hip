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

// Return the p-th percentile (p in [0,1]) of an already-sorted vector.
//
// We use a LOW percentile (p10) rather than the median to estimate the
// best-case, least-contended per-launch latency. Environmental noise on a
// shared CI runner (context switches, driver/JIT hiccups, memory pressure)
// only ever ADDS time to a launch, so a transient slow window inflates the
// median or max but not a low percentile. A genuine O(N) launch-overhead
// regression, by contrast, cannot be skipped -- every launch pays it -- so
// it raises even the p10 proportionally and the test still catches it.
static double percentile(const std::vector<double> &sorted, double p) {
  size_t idx = static_cast<size_t>(p * (sorted.size() - 1));
  return sorted[idx];
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
  return percentile(samples, 0.10);
}

// Maximum ratio of any measurement to the 0-stream baseline before the
// test fails.  The fixed implementation shows ~1x flat; the broken one
// shows 6-45x at 8+ idle streams.  With the best-of-SWEEPS p10 measurement
// above, a ratio near 1x is expected even on runners with a tiny sub-2us
// baseline (e.g. rusticl); under a fully CPU-saturated host the worst
// observed ratio was ~2.8x.  4x keeps comfortable margin against residual
// jitter while still cleanly catching the >=6x O(N) regression.
static const double SCALE_FAIL_THRESHOLD = 4.0;

int main(int argc, char **argv) {
  bool csv = (argc > 1 && strcmp(argv[1], "--csv") == 0);

  // Stream counts to sweep.
  const int counts[] = {0, 1, 2, 4, 8, 16, 32, 64};
  const int N        = sizeof(counts) / sizeof(counts[0]);
  const int WARMUP   = 20;
  const int ITERS    = 100;
  // Repeat the whole sweep several times and keep the minimum p10 per point.
  // Each point is thus measured in SWEEPS independent time windows spread
  // across the run. A transient system-load window (the dominant source of
  // flakiness on shared CI runners) inflates one sweep's p10 but not the
  // others, so the per-point minimum reflects the least-contended window --
  // this defeats the cross-window bias that a single sweep suffers when the
  // machine happens to be busy while the high-stream-count points (measured
  // last) are timed. A genuine O(N) launch-overhead regression is present in
  // every sweep, so the minimum still exposes it.
  const int SWEEPS   = 7;

  if (!csv) {
    printf("hipLaunchKernel overhead benchmark  (%d samples x %d sweeps per point)\n",
           ITERS, SWEEPS);
    printf("Null-stream launch + sync while N idle blocking streams exist\n\n");
    printf("  Idle streams | p10 ns/launch (best of %d sweeps)\n", SWEEPS);
    printf("  -------------|------------------\n");
  } else {
    printf("idle_streams,p10_ns_per_launch\n");
  }

  double results[N];
  for (int i = 0; i < N; ++i)
    results[i] = -1.0;
  for (int s = 0; s < SWEEPS; ++s) {
    for (int i = 0; i < N; ++i) {
      double v = measureLaunchNs(counts[i], WARMUP, ITERS);
      if (results[i] < 0.0 || v < results[i])
        results[i] = v;
    }
  }

  double base = results[0];
  for (int i = 0; i < N; ++i) {
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
