#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "hip/hip_runtime.h"

#define CHECK(cmd)                                                             \
  do {                                                                         \
    hipError_t e = (cmd);                                                      \
    if (e != hipSuccess) {                                                     \
      std::cerr << "HIP error " << hipGetErrorString(e) << " at " << __FILE__ \
                << ":" << __LINE__ << "\n";                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static constexpr int N_ITERS = 1000;
static constexpr size_t BUF_SIZE = 1024 * 1024 * 4; // 4 MB

int main() {
  CHECK(hipSetDevice(0));

  double *HostBuf = (double *)malloc(BUF_SIZE);
  memset(HostBuf, 0, BUF_SIZE);

  double *DevBuf;
  CHECK(hipMalloc(&DevBuf, BUF_SIZE));

  auto bench = [&](const char *label) {
    std::vector<double> timings(N_ITERS);
    CHECK(hipDeviceSynchronize());
    for (int i = 0; i < N_ITERS; ++i) {
      auto t0 = std::chrono::high_resolution_clock::now();
      CHECK(hipMemcpy(DevBuf, HostBuf, BUF_SIZE, hipMemcpyHostToDevice));
      CHECK(hipDeviceSynchronize());
      auto t1 = std::chrono::high_resolution_clock::now();
      timings[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double total = std::accumulate(timings.begin(), timings.end(), 0.0);
    double avg   = total / N_ITERS;
    double mn    = *std::min_element(timings.begin(), timings.end());
    double mx    = *std::max_element(timings.begin(), timings.end());
    // Use the MINIMUM for the pass/fail comparison. A 4 MB H2D copy is
    // bandwidth-bound, so its fastest observed time is a true physical floor:
    // OS-scheduling noise and the cold-start of whichever bench runs first
    // can only ADD time (see how avg/median/max blow up while min stays put),
    // so both the average and the median make the with/without ratio flaky on
    // shared CI runners -- taking the min removes the need for a warmup pass.
    // hipHostRegister adding a real per-copy cost would raise even the floor,
    // so the minimum still catches a genuine regression.
    std::sort(timings.begin(), timings.end());
    double med   = timings[N_ITERS / 2];
    std::cout << N_ITERS << " hipMemcpy H2D (" << BUF_SIZE / (1024 * 1024)
              << " MB) " << label << ": "
              << "median=" << med << " ms  avg=" << avg << " ms  min=" << mn
              << " ms  max=" << mx << " ms\n";
    return mn;
  };

  CHECK(hipHostRegister(HostBuf, BUF_SIZE, 0));
  double min_with = bench("WITH    hipHostRegister");
  CHECK(hipHostUnregister(HostBuf));

  double min_without = bench("WITHOUT hipHostRegister");

  std::cout << "overhead factor (min): " << min_with / min_without << "x\n";

  free(HostBuf);
  CHECK(hipFree(DevBuf));
// hipHostRegister only speeds copies up on datacenter GPUs (~2x on Aurora/PVC
// via zexDriverImportExternalPointer). On client Arc (Alchemist/Battlemage) the
// same import genuinely makes H2D copies ~13% SLOWER -- a real, order-independent
// driver behavior, confirmed with a pure Level Zero reproducer across PVC/A380/
// B570 (see CHIP-SPV/chipStar#1324), not a chipStar or measurement artifact. So a
// modest slowdown is expected here; only fail on a pathological regression (>50%).
  if (min_with / min_without > 1.5) {
    printf("Fail\n");
    return 1;
  }
  printf("Pass\n");
  return 0;
}
