// bf16 __shfl*_sync with the width omitted must use warpSize, not 32
// (CHIP-SPV/chipStar#1763). Only fails on builds where warpSize != 32.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdio>

__global__ void shfl(float *Out) {
  int L = threadIdx.x;
  __hip_bfloat16 V = __float2bfloat16((float)L);
  Out[L] = __bfloat162float(__shfl_sync(~0u, V, warpSize + 1));
  Out[warpSize + L] = __bfloat162float(__shfl_xor_sync(~0u, V, warpSize));
  Out[2 * warpSize + L] = __bfloat162float(__shfl_down_sync(~0u, V, 1));
}

int main() {
  hipDeviceProp_t Props;
  (void)hipGetDeviceProperties(&Props, 0);
  int W = Props.warpSize;
  float *D = nullptr;
  (void)hipMalloc(&D, 3 * W * sizeof(float));
  shfl<<<1, W>>>(D);
  float H[3 * 64];
  (void)hipMemcpy(H, D, 3 * W * sizeof(float), hipMemcpyDeviceToHost);
  (void)hipFree(D);

  int Bad = 0;
  for (int L = 0; L < W; L++) {
    float Want[3] = {1.0f, (float)L, (float)(L == W - 1 ? L : L + 1)};
    for (int K = 0; K < 3; K++)
      if (H[K * W + L] != Want[K]) {
        printf("FAIL: op %d lane %d got %f want %f\n", K, L, H[K * W + L],
               Want[K]);
        Bad++;
      }
  }
  if (!Bad)
    printf("PASSED\n");
  return Bad ? 1 : 0;
}
