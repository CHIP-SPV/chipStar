// An out-of-range CHIP_PLATFORM (set by ctest) must fail initialization, not crash.
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  int Count = 0;
  hipError_t Err = hipGetDeviceCount(&Count);
  if (Err != hipErrorInitializationError) {
    printf("FAIL: hipGetDeviceCount returned %s\n", hipGetErrorName(Err));
    return 1;
  }
  printf("PASS\n");
  return 0;
}
