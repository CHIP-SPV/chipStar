#include <stdio.h>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"


extern __global__ void saxpy();

  dim3 blocks 	    = {1,1,1};
  dim3 threads      = {1,1,1};

int main(void) {
  hipLaunchKernel((const void*)(&saxpy), blocks, threads, 0, 0, 0);
  hipDeviceSynchronize();
  return 0;
}