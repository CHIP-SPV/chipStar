/**
 * @file TestHipMemcpyInvalidPtr.hip
 * @brief Regression test for https://github.com/CHIP-SPV/chipStar/issues/1223
 *
 * When hipMalloc fails (e.g. OOM) and the application ignores the error,
 * calling hipMemcpy with the uninitialised destination pointer must return
 * an error — not wedge the process.
 */
#include <hip/hip_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
  // Query device max single allocation size so we can request more.
  hipDeviceProp_t Props;
  hipError_t Err = hipGetDeviceProperties(&Props, 0);
  if (Err != hipSuccess) {
    fprintf(stderr, "hipGetDeviceProperties failed: %d\n", (int)Err);
    return 1;
  }

  // Request twice the reported total memory — guaranteed to fail.
  size_t ReqBytes = (size_t)Props.totalGlobalMem * 2;
  fprintf(stderr, "Requesting %zu bytes (device total = %zu)\n",
          ReqBytes, (size_t)Props.totalGlobalMem);

  char *D = (char *)0xdeadbeef;
  Err = hipMalloc((void **)&D, ReqBytes);
  fprintf(stderr, "hipMalloc returned %d (%s)\n", (int)Err,
          hipGetErrorString(Err));
  assert(Err != hipSuccess && "hipMalloc should have failed");

  // Small host buffer — we only need it to be a valid source pointer.
  const size_t HostSize = 64;
  char H[HostSize];
  memset(H, 0xab, HostSize);

  // hipMemcpy with the invalid device pointer must return an error,
  // NOT wedge the process.
  fprintf(stderr, "hipMemcpy(D=%p, H, %zu, H2D)\n", (void *)D, HostSize);
  Err = hipMemcpy(D, H, HostSize, hipMemcpyHostToDevice);
  fprintf(stderr, "hipMemcpy returned %d (%s)\n", (int)Err,
          hipGetErrorString(Err));
  assert(Err != hipSuccess && "hipMemcpy with invalid dst must fail");

  // Also test Device-to-Host direction.
  fprintf(stderr, "hipMemcpy(H, D=%p, %zu, D2H)\n", (void *)D, HostSize);
  Err = hipMemcpy(H, D, HostSize, hipMemcpyDeviceToHost);
  fprintf(stderr, "hipMemcpy returned %d (%s)\n", (int)Err,
          hipGetErrorString(Err));
  assert(Err != hipSuccess && "hipMemcpy with invalid src must fail");

  fprintf(stderr, "PASSED\n");
  return 0;
}
