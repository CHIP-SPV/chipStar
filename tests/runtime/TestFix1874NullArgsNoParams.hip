// Reproduces #1874: hipLaunchKernel rejects Args == nullptr for a kernel
// without parameters; null Args must still be rejected when it has some.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void noParams() {}
__global__ void oneParam(int *P) { *P = 1; }

int main() {
  hipError_t Err = hipLaunchKernel((const void *)noParams, dim3(1), dim3(1),
                                   nullptr, 0, 0);
  if (Err != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
    printf("FAIL: zero-parameter launch with null Args returned %s\n",
           hipGetErrorName(Err));
    return 1;
  }
  Err = hipLaunchKernel((const void *)oneParam, dim3(1), dim3(1), nullptr, 0,
                        0);
  if (Err != hipErrorInvalidValue) {
    printf("FAIL: one-parameter launch with null Args returned %s\n",
           hipGetErrorName(Err));
    return 1;
  }
  (void)hipGetLastError();
  printf("PASSED\n");
  return 0;
}
