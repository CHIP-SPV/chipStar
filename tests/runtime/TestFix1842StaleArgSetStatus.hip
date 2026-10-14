// Reproduces #1842: after a launch fails in the driver, launching a kernel
// whose handle is reused from an earlier launch reports that stale error.

#include <hip/hip_runtime.h>
#include "backend/OpenCL/CHIPBackendOpenCL.hh"

#include <cstdio>

struct S {
  int A, B;
};

__global__ void k(S s, int *Out) { *Out = s.A + s.B; }
__global__ void other(S s, int *Out) { *Out = s.A; }

static hipError_t launchK(S Arg, int *Out, int *Res) {
  hipLaunchKernelGGL(k, dim3(1), dim3(1), 0, 0, Arg, Out);
  hipError_t Err = hipGetLastError();
  if (Err == hipSuccess)
    Err = hipMemcpy(Res, Out, sizeof(int), hipMemcpyDeviceToHost);
  return Err;
}

int main() {
  int *Out, Res = 0;
  if (hipMalloc(&Out, sizeof(int)) != hipSuccess) {
    std::printf("FAIL: hipMalloc\n");
    return 1;
  }
  // Level Zero aborts on a zero block dimension (#1843).
  if (!dynamic_cast<CHIPBackendOpenCL *>(::Backend)) {
    std::printf("HIP_SKIP_THIS_TEST: not running on OpenCL\n");
    return CHIP_SKIP_TEST;
  }
  // Builds k and leaves its handle in the pool for reuse.
  hipError_t Err = launchK(S{2, 3}, Out, &Res);
  if (Err != hipSuccess || Res != 5) {
    std::printf("FAIL: first launch: %s, result %d\n", hipGetErrorString(Err),
                Res);
    return 1;
  }
  // A (0,1,1) block fails in clEnqueueNDRangeKernel on Intel; PoCL accepts it.
  hipLaunchKernelGGL(other, dim3(1), dim3(0, 1, 1), 0, 0, S{2, 3}, Out);
  if (hipGetLastError() == hipSuccess) {
    std::printf("HIP_SKIP_THIS_TEST: the (0,1,1) block launch did not fail\n");
    return CHIP_SKIP_TEST;
  }
  Err = launchK(S{4, 5}, Out, &Res);
  if (Err != hipSuccess || Res != 9) {
    std::printf("FAIL: launch after a failed one: %s, result %d\n",
                hipGetErrorString(Err), Res);
    return 1;
  }
  (void)hipFree(Out);
  std::printf("PASS\n");
  return 0;
}
