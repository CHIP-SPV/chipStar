// #2025: with CHIP_LAZY_JIT=0 a failed module build must surface at launch.
#include <hip/hip_runtime.h>
#include <cstdio>

// Declared but never defined: the driver rejects the build or drops the kernel.
extern __device__ void foo(int);
__global__ void kernel(int x) { foo(x); }

int main() {
  kernel<<<1, 1>>>(123);
  hipError_t Err = hipGetLastError();
  // Accept a module build or link error.
  if (Err == hipErrorSharedObjectInitFailed || Err == hipErrorInvalidImage) {
    printf("PASSED\n");
    return 0;
  }
  printf("FAILED: got %s\n", hipGetErrorName(Err));
  return 1;
}
