// #2025: with CHIP_LAZY_JIT=0 a failed module build must surface at launch.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

// Declared but never defined, so the device program build fails at runtime.
extern __device__ void foo(int);
__global__ void kernel(int x) { foo(x); }

int main() {
  hipDeviceProp_t Props;
  if (hipGetDeviceProperties(&Props, 0) == hipSuccess &&
      strstr(Props.name, "Mali")) {
    printf("HIP_SKIP_THIS_TEST: Mali accepts the module, see #2026\n");
    return CHIP_SKIP_TEST;
  }
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
