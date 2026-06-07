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
    std::cout << N_ITERS << " hipMemcpy H2D (" << BUF_SIZE / (1024 * 1024)
              << " MB) " << label << ": "
              << "avg=" << avg << " ms  min=" << mn << " ms  max=" << mx
              << " ms\n";
    return avg;
  };

  CHECK(hipHostRegister(HostBuf, BUF_SIZE, 0));
  double avg_with = bench("WITH    hipHostRegister");
  CHECK(hipHostUnregister(HostBuf));

  double avg_without = bench("WITHOUT hipHostRegister");

  std::cout << "overhead factor (avg): " << avg_with / avg_without << "x\n";

  free(HostBuf);
  CHECK(hipFree(DevBuf));
// with hiphost register it should be faster, but as long as it's not slower
// it's ok
  if (avg_with / avg_without > 1.1) {
    printf("Fail\n");
    return 1;
  }
  printf("Pass\n");
  return 0;
}
