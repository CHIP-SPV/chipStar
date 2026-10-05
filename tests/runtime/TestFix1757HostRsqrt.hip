// Reproduces https://github.com/CHIP-SPV/chipStar/issues/1757: the host
// OCML class test never matches, so host rsqrt(2.0) skips its refinement
// step and is 1 ulp off.

#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  volatile double X = 2.0;
  double R = rsqrt(X);
  // 1/sqrt(2) = 0x1.6a09e667f3bcc908...p-1, which rounds up.
  if (R != 0x1.6a09e667f3bcdp-1) {
    printf("FAILED: rsqrt(2.0) = %a, expected 0x1.6a09e667f3bcdp-1\n", R);
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
