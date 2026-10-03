// __all and __any must vote only over the lanes a warp has: a 12-thread block
// and the partial last warp of a 44-thread block (CHIP-SPV/chipStar#1735).
#include "CHIPDriver.hh"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

__global__ void votes(int *Out) {
  int T = threadIdx.x;
  Out[T] = __all(1) | (__any(T == 0) << 1) | (__all(T != 0) << 2);
}

int main() {
  hipDeviceProp_t Props;
  (void)hipGetDeviceProperties(&Props, 0);
  // Mali cannot link the ballot device library (CHIP-SPV/chipStar#1368).
  if (!Props.arch.hasWarpBallot || std::strstr(Props.name, "Mali")) {
    printf("HIP_SKIP_THIS_TEST: device cannot run __ballot()\n");
    return CHIP_SKIP_TEST;
  }
  int Bad = 0;
  for (int N : {12, 44}) {
    int *D, H[44];
    (void)hipMalloc(&D, sizeof(H));
    votes<<<1, N>>>(D);
    (void)hipMemcpy(H, D, N * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(D);
    for (int T = 0; T < N; ++T) {
      bool First = T < Props.warpSize;
      int Exp = 1 | (First << 1) | (!First << 2);
      if (H[T] != Exp && Bad++ < 4)
        printf("block %d lane %d: got %d, expected %d\n", N, T, H[T], Exp);
    }
  }
  printf("%s\n", Bad ? "FAILED" : "PASSED");
  return Bad != 0;
}
