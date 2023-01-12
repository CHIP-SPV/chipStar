#include <stdio.h>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

extern __global__ void saxpy();

dim3 Blocks = {1, 1, 1};
dim3 Threads = {1, 1, 1};
int* Args;

int main(void) {
  hipLaunchKernel((const void *)(&saxpy), Blocks, Threads, (void**)&Args, 0, 0);
  hipDeviceSynchronize();
  return 0;
}
