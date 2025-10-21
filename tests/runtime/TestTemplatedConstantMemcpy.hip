// Reproducer for Issue #977: hipMemcpyToSymbol with templated __constant__ memory
#include <hip/hip_runtime.h>
#include <cstdio>

#define MAX_MASK_WIDTH 10

template <typename T>
__constant__ T mask[MAX_MASK_WIDTH];

template <typename T>
static int runTestOnce() {
  T hostMask[MAX_MASK_WIDTH] = {};
  hipError_t st = hipMemcpyToSymbol(mask<T>, hostMask, 3 * sizeof(T));
  if (st != hipSuccess) {
    std::printf("FAIL: hipMemcpyToSymbol error=%d\n", static_cast<int>(st));
    return 1;
  }
  st = hipDeviceSynchronize();
  if (st != hipSuccess) {
    std::printf("FAIL: hipDeviceSynchronize error=%d\n", static_cast<int>(st));
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}

int main() {
  // Minimal device presence check
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) {
    std::puts("SKIP: No HIP device available");
    return 77; // CHIP_SKIP_TEST
  }

  // The issue was reported with double; use double to maximize chances to repro
  return runTestOnce<double>();
}


