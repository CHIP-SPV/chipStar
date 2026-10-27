#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>
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

static constexpr int N_ROUNDS = 10;
static constexpr int N_ITERS = 100;
static constexpr size_t BUF_SIZE = 1024 * 1024 * 4; // 4 MB

int main() {
  CHECK(hipSetDevice(0));

  double *HostBuf = (double *)malloc(BUF_SIZE);
  memset(HostBuf, 0, BUF_SIZE);

  double *DevBuf;
  CHECK(hipMalloc(&DevBuf, BUF_SIZE));

  auto bench = [&]() {
    std::vector<double> t(N_ITERS);
    CHECK(hipDeviceSynchronize());
    for (int i = 0; i < N_ITERS; ++i) {
      auto t0 = std::chrono::high_resolution_clock::now();
      CHECK(hipMemcpy(DevBuf, HostBuf, BUF_SIZE, hipMemcpyHostToDevice));
      CHECK(hipDeviceSynchronize());
      auto t1 = std::chrono::high_resolution_clock::now();
      t[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(t.begin(), t.end());
    return t[N_ITERS / 2];
  };

  // Compare WITH/WITHOUT within each short round; a load change then skews at most one round's ratio.
  std::vector<double> ratios;
  for (int r = 0; r < N_ROUNDS; ++r) {
    CHECK(hipHostRegister(HostBuf, BUF_SIZE, 0));
    double with = bench();
    CHECK(hipHostUnregister(HostBuf));
    double without = bench();
    ratios.push_back(with / without);
    std::cout << "round " << r << ": median WITH=" << with
              << " ms  WITHOUT=" << without << " ms  ratio=" << with / without
              << "\n";
  }
  std::sort(ratios.begin(), ratios.end());
  double ratio = ratios[N_ROUNDS / 2];
  std::cout << "overhead factor (median of per-round ratios): " << ratio << "x\n";

  free(HostBuf);
  CHECK(hipFree(DevBuf));
// hipHostRegister only speeds copies up on datacenter GPUs (~2x on Aurora/PVC
// via zexDriverImportExternalPointer). On client Arc (Alchemist/Battlemage) the
// same import genuinely makes H2D copies ~13% SLOWER -- a real, order-independent
// driver behavior, confirmed with a pure Level Zero reproducer across PVC/A380/
// B570 (see CHIP-SPV/chipStar#1324), not a chipStar or measurement artifact. So a
// modest slowdown is expected here; only fail on a pathological regression (>30%).
  if (ratio > 1.3) {
    printf("Fail\n");
    return 1;
  }
  printf("Pass\n");
  return 0;
}
