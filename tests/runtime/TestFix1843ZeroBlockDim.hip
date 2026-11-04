// Reproduces #1843: a zero block dimension aborts on Level Zero instead of
// returning hipErrorInvalidConfiguration.

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void k() {}

int main() {
  const dim3 Blocks[] = {dim3(0, 1, 1), dim3(1, 0, 1), dim3(1, 1, 0)};
  int Failed = 0;
  for (const dim3 &Block : Blocks) {
    hipLaunchKernelGGL(k, dim3(1), Block, 0, 0);
    hipError_t Err = hipGetLastError();
    if (Err != hipErrorInvalidConfiguration) {
      printf("FAIL: block (%u,%u,%u) returned %s\n", Block.x, Block.y, Block.z,
             hipGetErrorName(Err));
      Failed = 1;
    }
  }
  if (hipDeviceSynchronize() != hipSuccess) {
    printf("FAIL: hipDeviceSynchronize\n");
    Failed = 1;
  }
  if (!Failed)
    printf("PASSED\n");
  return Failed;
}
