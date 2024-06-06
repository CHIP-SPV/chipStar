#include <cstdio>
#include <cuda_runtime.h>

extern __global__ void world();
__global__ void hello() { printf("Hello"); }

int main() {
  hello<<<1, 1>>>();
  world<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
