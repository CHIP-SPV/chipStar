// https://github.com/CHIP-SPV/chipStar/issues/2281: a failing
// clCreateCommandQueueWithProperties must come back as a HIP error, not abort.
#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
  uintptr_t Handles[8];
  int N = 0;
  if (hipGetBackendNativeHandles(0, nullptr, &N) != hipSuccess || N < 3 ||
      N > 8 || hipGetBackendNativeHandles(0, Handles, nullptr) != hipSuccess) {
    printf("FAIL: hipGetBackendNativeHandles\n");
    return 1;
  }
  if (strcmp(reinterpret_cast<const char *>(Handles[0]), "opencl") != 0) {
    printf("SKIP: not the OpenCL backend\n");
    return 77; // CHIP_SKIP_TEST
  }
  uintptr_t Args[4] = {Handles[1], Handles[2], 0, 0}; // NULL cl_context
  int Err = hipInitFromNativeHandles(Args, 4);
  if (Err != hipErrorInvalidResourceHandle) {
    printf("FAIL: got %d, want hipErrorInvalidResourceHandle\n", Err);
    return 1;
  }
  printf("PASS\n");
  fflush(stdout);
  // Exit handlers crash on the half-built Backend until
  // https://github.com/CHIP-SPV/chipStar/pull/2278 lands.
  std::_Exit(0);
}
