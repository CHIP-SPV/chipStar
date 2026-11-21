// hipStreamPerThread must work on the replacement Backend after
// hipInitFromNativeHandles.
// https://github.com/CHIP-SPV/chipStar/issues/2282
#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>
#include "ze_api.h"
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
  if (hipStreamSynchronize(hipStreamPerThread) != hipSuccess) {
    printf("FAIL: first hipStreamSynchronize\n");
    return 1;
  }

  uintptr_t h[8] = {};
  int n = 0;
  (void)hipGetBackendNativeHandles(0, nullptr, &n);
  if (n < 4 || n > 8) {
    printf("HIP_SKIP_THIS_TEST: unexpected handle count %d\n", n);
    return CHIP_SKIP_TEST;
  }
  (void)hipGetBackendNativeHandles(0, h, nullptr);
  if (strcmp(reinterpret_cast<const char *>(h[0]), "level0") != 0) {
    printf("HIP_SKIP_THIS_TEST: not the level0 backend\n");
    return CHIP_SKIP_TEST;
  }

  ze_context_handle_t Ctx;
  ze_context_desc_t CtxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  if (zeContextCreate(reinterpret_cast<ze_driver_handle_t>(h[1]), &CtxDesc,
                      &Ctx) != ZE_RESULT_SUCCESS) {
    printf("FAIL: zeContextCreate\n");
    return 1;
  }
  // driver, device, app-owned context, null queue (chipStar makes its own)
  uintptr_t Native[4] = {h[1], h[2], reinterpret_cast<uintptr_t>(Ctx), 0};
  if (hipInitFromNativeHandles(Native, 4) != hipSuccess) {
    printf("FAIL: hipInitFromNativeHandles\n");
    return 1;
  }

  unsigned char *D = nullptr, Out[64];
  if (hipMalloc(&D, sizeof(Out)) != hipSuccess ||
      hipMemsetAsync(D, 0x5a, sizeof(Out), hipStreamPerThread) != hipSuccess ||
      hipStreamSynchronize(hipStreamPerThread) != hipSuccess ||
      hipMemcpy(Out, D, sizeof(Out), hipMemcpyDeviceToHost) != hipSuccess) {
    printf("FAIL: memset on hipStreamPerThread\n");
    return 1;
  }
  for (unsigned char B : Out)
    if (B != 0x5a) {
      printf("FAIL: wrong byte %d\n", B);
      return 1;
    }
  printf("PASS\n");
  return 0;
}
