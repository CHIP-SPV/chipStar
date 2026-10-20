#include <cstdio>
#include <cuda_runtime.h>

__device__ int twice(int x);

__global__ void k(int *out) { *out = twice(21); }

int main() {
  int *d, h = 0;
  cudaMalloc(&d, sizeof(int));
  k<<<1, 1>>>(d);
  cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d);
  if (h != 42) {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
