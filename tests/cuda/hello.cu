#include <cuda_runtime.h>
__global__ void k() { printf("Hello, World!\n"); }
int main() {
  k<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
