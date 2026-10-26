// A double __shfl_xor added back into its own operand must give in[i] + in[i^1]
// and not hang under IGC fp64 emulation (CHIP-SPV/chipStar#2106).

#include <hip/hip_runtime.h>

#include <cstdio>

__global__ void k(const double *in, double *out) {
  double s = in[threadIdx.x];
  s += __shfl_xor(s, 1);
  out[threadIdx.x] = s;
}

int main() {
  double h[32], r[32], *in, *out;
  for (int i = 0; i < 32; i++)
    h[i] = i + 0.5;
  if (hipMalloc(&in, sizeof h) || hipMalloc(&out, sizeof h) ||
      hipMemcpy(in, h, sizeof h, hipMemcpyHostToDevice)) {
    printf("FAIL: setup\n");
    return 1;
  }
  k<<<1, 32>>>(in, out);
  if (hipMemcpy(r, out, sizeof r, hipMemcpyDeviceToHost)) {
    printf("FAIL: copy back\n");
    return 1;
  }
  for (int i = 0; i < 32; i++)
    if (r[i] != h[i] + h[i ^ 1]) {
      printf("FAIL: lane %d got %g expected %g\n", i, r[i], h[i] + h[i ^ 1]);
      return 1;
    }
  printf("PASSED\n");
  return 0;
}
