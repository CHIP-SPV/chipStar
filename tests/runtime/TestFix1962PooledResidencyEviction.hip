// Another Level Zero context in the process probing a pooled allocation
// (alloc, make resident, free) must not leave chipStar's allocations non-resident.
// https://github.com/CHIP-SPV/chipStar/issues/1962
#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>
#include "ze_api.h"
#include <cstdint>
#include <cstdio>
#include <cstring>

struct S {
  double *p;
  double v;
};

__global__ void fillS(S s, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    s.p[i] = s.v;
}

int main() {
  double *d1 = nullptr, *d2 = nullptr;
  if (hipMalloc(&d1, 256 * sizeof(double)) != hipSuccess) {
    printf("FAIL: hipMalloc failed\n");
    return 1;
  }
  fillS<<<1, 256>>>(S{d1, 1.0}, 256);
  (void)hipDeviceSynchronize();

  int n = 0;
  (void)hipGetBackendNativeHandles(0, nullptr, &n);
  uintptr_t h[8] = {};
  if (n < 4 || n > 8) {
    printf("HIP_SKIP_THIS_TEST: unexpected handle count %d\n", n);
    return CHIP_SKIP_TEST;
  }
  (void)hipGetBackendNativeHandles(0, h, nullptr);
  if (strcmp(reinterpret_cast<const char *>(h[0]), "level0") != 0) {
    printf("HIP_SKIP_THIS_TEST: not the level0 backend\n");
    return CHIP_SKIP_TEST;
  }
  auto drv = reinterpret_cast<ze_driver_handle_t>(h[1]);
  auto dev = reinterpret_cast<ze_device_handle_t>(h[2]);

  ze_context_handle_t c2;
  ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_device_mem_alloc_desc_t ad = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  void *p;
  if (zeContextCreate(drv, &cd, &c2) != ZE_RESULT_SUCCESS ||
      zeMemAllocDevice(c2, &ad, 1, 64, dev, &p) != ZE_RESULT_SUCCESS ||
      zeContextMakeMemoryResident(c2, dev, p, 1) != ZE_RESULT_SUCCESS ||
      zeMemFree(c2, p) != ZE_RESULT_SUCCESS) {
    printf("FAIL: Level Zero probe sequence failed\n");
    return 1;
  }

  if (hipMalloc(&d2, 256 * sizeof(double)) != hipSuccess) {
    printf("FAIL: hipMalloc after the probe failed\n");
    return 1;
  }
  fillS<<<1, 256>>>(S{d2, 3.0}, 256);
  if (hipDeviceSynchronize() != hipSuccess) {
    printf("FAIL: kernel did not complete\n");
    return 1;
  }
  double r[256];
  if (hipMemcpy(r, d2, sizeof(r), hipMemcpyDeviceToHost) != hipSuccess) {
    printf("FAIL: hipMemcpy failed\n");
    return 1;
  }
  for (double x : r)
    if (x != 3.0) {
      printf("FAIL: wrong value %f\n", x);
      return 1;
    }
  printf("PASS\n");
  return 0;
}
