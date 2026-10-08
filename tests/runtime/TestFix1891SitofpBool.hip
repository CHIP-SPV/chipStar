// Reproduces #1891: at -O2 LLVM folds (c ? -1 : 0) stored to float into
// sitofp i1, which the SPIR-V producers lower to +1.0 instead of -1.0.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void k(const float *In, float *Out) {
  int I = threadIdx.x;
  Out[I] = (In[I] == 0 ? -1 : 0);
}

int main() {
  float HIn[4] = {0, 1, 0, 2}, HOut[4], *DIn, *DOut;
  (void)hipMalloc(&DIn, sizeof HIn);
  (void)hipMalloc(&DOut, sizeof HOut);
  (void)hipMemcpy(DIn, HIn, sizeof HIn, hipMemcpyHostToDevice);
  k<<<1, 4>>>(DIn, DOut);
  (void)hipMemcpy(HOut, DOut, sizeof HOut, hipMemcpyDeviceToHost);
  int Bad = 0;
  for (int I = 0; I < 4; I++) {
    float Ref = (HIn[I] == 0 ? -1 : 0);
    printf("Out[%d] = %g (expected %g)\n", I, HOut[I], Ref);
    Bad += HOut[I] != Ref;
  }
  printf(Bad ? "FAIL\n" : "PASS\n");
  return Bad != 0;
}
