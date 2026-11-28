#include <hip/hip_runtime.h>

__global__ void syncTest(int *out) {
  const int tid = threadIdx.x;

  if (tid > 10)
    return;

  __syncthreads();
  out[tid] = tid + 1;
}

int main() {
  int *d, h[100] = {};
  hipMalloc(&d, sizeof h);
  hipMemset(d, 0, sizeof h);
  hipLaunchKernelGGL(syncTest, dim3(1), dim3(100), 0, 0, d);
  hipMemcpy(h, d, sizeof h, hipMemcpyDeviceToHost);
  hipFree(d);
  for (int i = 0; i < 100; i++)
    if (h[i] != (i <= 10 ? i + 1 : 0)) {
      printf("FAILED: out[%d] = %d\n", i, h[i]);
      return 1;
    }
  printf("PASSED\n");
  return 0;
}
