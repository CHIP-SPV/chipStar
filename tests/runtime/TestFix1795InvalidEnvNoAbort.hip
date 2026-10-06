// An invalid CHIP_* environment (set by ctest) must fail every HIP call, not abort at load.
#include <hip/hip_runtime.h>
#include <cstdio>

// Registers a fat binary, whose static initializer also hits the invalid environment.
__global__ void kernel() {}

int main() {
  for (int I = 0; I < 2; I++) {
    int Count = 0;
    hipError_t Err = hipGetDeviceCount(&Count);
    if (Err != hipErrorInitializationError) {
      printf("FAIL: hipGetDeviceCount call %d returned %s\n", I,
             hipGetErrorName(Err));
      return 1;
    }
  }
  printf("PASS\n");
  return 0;
}
