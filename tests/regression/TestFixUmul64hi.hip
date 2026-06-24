// Regression test for: __umul64hi and __mul64hi must return the high 64 bits
// of the 128-bit product, not 0.
//
// Bug: bitcode/devicelib.cl implemented __chip_umul64hi as (x*y)>>64 where x*y
// is a 64-bit truncated product.  Shifting a 64-bit value by 64 is undefined
// in SPIR-V (shift count >= bit width), causing IGC to crash with
// "Termination request" during JIT compilation of any kernel that uses it.
//
// Fix: use OpenCL's mul_hi(ulong, ulong) which computes the high 64 bits of
// the 128-bit product without undefined behavior.
//
// Pattern from HeCBench merkle-hip: ff_p_mult() calls __umul64hi() and
// any kernel using ff_p_mult() causes IGC to terminate during JIT.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

__global__ void testUmul64hi(uint64_t *results) {
  // 2^63 * 2 = 2^64 → high 64 bits = 1
  results[0] = __umul64hi((uint64_t)1 << 63, (uint64_t)2);
  // 0xFFFFFFFFFFFFFFFF * 0xFFFFFFFFFFFFFFFF
  // = 0xFFFFFFFFFFFFFFFE_0000000000000001
  // high 64 bits = 0xFFFFFFFFFFFFFFFE
  results[1] = __umul64hi(UINT64_MAX, UINT64_MAX);
  // 2^32 * 2^32 = 2^64 → high 64 bits = 1
  results[2] = __umul64hi((uint64_t)1 << 32, (uint64_t)1 << 32);
  // Any value * 0 = 0 → high 64 bits = 0
  results[3] = __umul64hi(UINT64_MAX, (uint64_t)0);
}

int main() {
  uint64_t *d_res;
  if (hipMalloc(&d_res, 4 * sizeof(uint64_t)) != hipSuccess) {
    printf("SKIP: no GPU\n");
    return 0;
  }
  hipMemset(d_res, 0, 4 * sizeof(uint64_t));
  hipLaunchKernelGGL(testUmul64hi, dim3(1), dim3(1), 0, 0, d_res);
  if (hipDeviceSynchronize() != hipSuccess) {
    printf("FAIL: kernel launch or sync failed\n");
    hipFree(d_res);
    return 1;
  }

  uint64_t res[4];
  hipMemcpy(res, d_res, sizeof(res), hipMemcpyDeviceToHost);
  hipFree(d_res);

  const uint64_t expected[4] = {
      (uint64_t)1,                    // (2^63 * 2) >> 64 = 1
      (uint64_t)0xFFFFFFFFFFFFFFFEULL, // (MAX*MAX) >> 64 = MAX-1
      (uint64_t)1,                    // (2^32 * 2^32) >> 64 = 1
      (uint64_t)0,                    // (MAX * 0) >> 64 = 0
  };

  int errs = 0;
  for (int i = 0; i < 4; i++) {
    if (res[i] != expected[i]) {
      printf("FAIL[%d]: got 0x%016llx want 0x%016llx\n", i,
             (unsigned long long)res[i], (unsigned long long)expected[i]);
      errs++;
    }
  }

  printf("%s\n", errs == 0 ? "PASS" : "FAIL");
  return errs ? 1 : 0;
}
