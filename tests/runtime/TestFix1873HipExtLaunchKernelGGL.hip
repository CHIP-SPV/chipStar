// Reproduces https://github.com/CHIP-SPV/chipStar/issues/1873:
// hipExtLaunchKernelGGL from hip/hip_ext.h does not compile on chipStar.

#include <hip/hip_ext.h>
#include <cstdio>

__global__ void k(int *Out, int V) { *Out = V; }

int main() {
  int *Out;
  int Host = 0;
  if (hipMalloc(&Out, sizeof(int)) != hipSuccess) {
    printf("FAILED: hipMalloc\n");
    return 1;
  }
  hipExtLaunchKernelGGL(k, dim3(1), dim3(1), 0, nullptr, nullptr, nullptr, 0,
                        Out, 42);
  if (hipMemcpy(&Host, Out, sizeof(int), hipMemcpyDeviceToHost) !=
          hipSuccess ||
      Host != 42) {
    printf("FAILED: Out = %d\n", Host);
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
