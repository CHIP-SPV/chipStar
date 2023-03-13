#include <hip/hip_runtime.h>
#include "common.h"

__global__ void Interp() {
  CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z;
  if (elem == 0) {
    printf("%d, %d, %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    __syncthreads();
  }
}

int main() {
  hipLaunchKernelGGL(&Interp, dim3(1, 1, 1), dim3(1, 4, 16), 0, 0);
}