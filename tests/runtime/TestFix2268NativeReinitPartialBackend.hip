// Regression test for https://github.com/CHIP-SPV/chipStar/issues/2268
// A failed hipInitFromNativeHandles must not leave a half built Backend behind.

#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>
#include "CHIPDriver.hh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(Cond)                                                            \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      printf("FAIL: %s\n", #Cond);                                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  int Count = 0;
  if (hipGetDeviceCount(&Count) != hipSuccess || Count < 1) {
    printf("HIP_SKIP_THIS_TEST: no device\n");
    return CHIP_SKIP_TEST;
  }

  int N = 0;
  CHECK(hipGetBackendNativeHandles(0, nullptr, &N) == hipSuccess);
  uintptr_t Handles[8] = {};
  CHECK(N >= 1 && N <= 8);
  CHECK(hipGetBackendNativeHandles(0, Handles, nullptr) == hipSuccess);
  if (strcmp(reinterpret_cast<const char *>(Handles[0]), "opencl") != 0) {
    printf("HIP_SKIP_THIS_TEST: OpenCL only trigger\n");
    return CHIP_SKIP_TEST;
  }

  // The first init already succeeded, so only the native reinit sees the bad strategy.
  setenv("CHIP_OCL_USE_ALLOC_STRATEGY", "bogus", 1);
  ChipEnvVars = EnvVars();

  uintptr_t Args[4] = {Handles[1], Handles[2], 0, 0};
  CHECK(hipInitFromNativeHandles(Args, 4) != hipSuccess);

  int C = 0, D = 0;
  void *P = nullptr;
  CHECK(hipGetDeviceCount(&C) != hipSuccess);
  CHECK(hipGetDevice(&D) != hipSuccess);
  CHECK(hipMalloc(&P, 64) != hipSuccess);

  printf("PASS\n");
  return 0;
}
