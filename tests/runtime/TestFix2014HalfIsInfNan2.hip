// Reproduces #2014: __hisinf2/__hisnan2 returned -1.0 instead of 1.0 for true lanes.
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>

__global__ void k(float *Out) {
  __half2 Inf = __floats2half2_rn(INFINITY, 1.0f);
  __half2 Nan = __floats2half2_rn(NAN, 1.0f);
  __half2 I = __hisinf2(Inf), N = __hisnan2(Nan);
  Out[0] = __low2float(I);
  Out[1] = __high2float(I);
  Out[2] = __low2float(N);
  Out[3] = __high2float(N);
}

int main() {
  float *D, H[4];
  if (hipMalloc(&D, sizeof(H)) != hipSuccess)
    return 1;
  k<<<1, 1>>>(D);
  if (hipGetLastError() == hipErrorSharedObjectInitFailed)
    return CHIP_SKIP_TEST; // Device cannot build half kernels (e.g. PoCL).
  if (hipMemcpy(H, D, sizeof(H), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  if (H[0] != 1 || H[1] != 0 || H[2] != 1 || H[3] != 0) {
    printf("FAIL: isinf2 {%g,%g} nan2 {%g,%g}, expected {1,0} {1,0}\n", H[0],
           H[1], H[2], H[3]);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
