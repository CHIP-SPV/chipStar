#include <hip/hip_runtime.h>

__device__ void* testNew() {
  void* i = operator new(1);
  return i;
}

extern "C" __global__ void kernel() {
  testNew();
  // int *k = static_cast<int*>(malloc(sizeof(int)));
}

int main() {
  hipLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, 0);
  hipDeviceSynchronize();
  return 0;
}