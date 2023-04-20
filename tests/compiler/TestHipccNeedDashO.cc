#include <hip/hip_runtime.h>

__global__ void kernel() {
  printf("Hello World!");
}

int main() {
  hipLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, 0);
  hipDeviceSynchronize();
  return 0;
}