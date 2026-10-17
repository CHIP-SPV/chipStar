// CHIP-SPV/chipStar#1851: a zero local work size must return
// hipErrorInvalidValue, not SIGFPE.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// libCHIP exports this with C++ linkage (#1857); include <hip/hip_ext.h> once
// fixed.
hipError_t hipExtModuleLaunchKernel(hipFunction_t, uint32_t, uint32_t,
                                    uint32_t, uint32_t, uint32_t, uint32_t,
                                    size_t, hipStream_t, void **, void **,
                                    hipEvent_t, hipEvent_t, uint32_t);

int main() {
  // Non-null sentinel; the zero-size check returns before it is used.
  hipFunction_t Kernel = reinterpret_cast<hipFunction_t>(uintptr_t{1});

  struct Case {
    const char *Name;
    uint32_t Lx, Ly, Lz;
  } Cases[] = {
      {"local x = 0", 0, 1, 1},
      {"local y = 0", 1, 0, 1},
      {"local z = 0", 1, 1, 0},
  };

  int Failures = 0;
  for (const Case &C : Cases) {
    hipError_t Err = hipExtModuleLaunchKernel(
        Kernel, 64, 64, 1, C.Lx, C.Ly, C.Lz, 0, nullptr, nullptr, nullptr,
        nullptr, nullptr, 0);
    if (Err != hipErrorInvalidValue) {
      printf("FAILED: %s -> %s, expected hipErrorInvalidValue\n", C.Name,
             hipGetErrorString(Err));
      ++Failures;
    }
  }

  if (Failures)
    return 1;
  printf("PASS\n");
  return 0;
}
