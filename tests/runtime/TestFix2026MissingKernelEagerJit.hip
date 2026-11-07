// #2026: a module lacking a registered kernel must fail as a module load error.
#include <hip/hip_runtime.h>
#include <cstdio>

// Registers first, so a module that failed to load must fail it too.
__global__ void good(int) {}

#ifndef __HIP_DEVICE_COMPILE__
// Host only, so the module builds on every driver but lacks this kernel.
__global__ void missing(int) {}

int main() {
  hipError_t Errs[3];
  good<<<1, 1>>>(0);
  Errs[0] = hipGetLastError();
  missing<<<1, 1>>>(0);
  Errs[1] = hipGetLastError();
  good<<<1, 1>>>(0);
  Errs[2] = hipGetLastError();
  for (hipError_t Err : Errs)
    if (Err != hipErrorSharedObjectInitFailed) {
      printf("FAILED: got %s\n", hipGetErrorName(Err));
      return 1;
    }
  printf("PASSED\n");
  return 0;
}
#endif
