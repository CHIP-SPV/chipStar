// Reproduces #1767: hipHostMalloc was charged against totalGlobalMem, so on a
// discrete GPU pinned host memory lowered hipMemGetInfo's free memory and made
// hipMalloc of memory the device still had fail with hipErrorOutOfMemory.
#include <hip/hip_runtime_api.h>
#include <cstdio>

int main() {
  int Integrated = 0;
  if (hipDeviceGetAttribute(&Integrated, hipDeviceAttributeIntegrated, 0) !=
      hipSuccess)
    return 1;
  if (Integrated)
    return CHIP_SKIP_TEST; // Host allocations do use device memory there.

  size_t Free0, Free1, Total;
  void *Host;
  if (hipMemGetInfo(&Free0, &Total) != hipSuccess ||
      hipHostMalloc(&Host, 1 << 20) != hipSuccess ||
      hipMemGetInfo(&Free1, &Total) != hipSuccess)
    return 1;
  if (hipHostFree(Host) != hipSuccess)
    return 1;
  if (Free1 != Free0) {
    printf("FAIL: hipHostMalloc of 1 MiB lowered free device memory by %zu\n",
           Free0 - Free1);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
