#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  // When CHIP_BE is not set, chipStar should pick a working backend
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  if (err != hipSuccess) {
    printf("FAIL: hipGetDeviceCount returned %d\n", err);
    return 1;
  }
  if (count < 1) {
    printf("FAIL: no devices found\n");
    return 1;
  }
  printf("PASS: found %d device(s) with default backend\n", count);
  return 0;
}
