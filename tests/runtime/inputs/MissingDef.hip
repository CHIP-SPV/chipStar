#include <hip/hip_runtime.h>

extern __device__ void foo(int);
__global__ void k() { foo(threadIdx.x + blockIdx.x * blockDim.x); }

int main() {
  k<<<64, 1>>>();
  hipDeviceSynchronize();
  return 0;
}
