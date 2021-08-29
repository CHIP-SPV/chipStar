#include "hipDeviceLink.h"

__global__ void Write(const int *in) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  global[tid] = in[tid];
}

void writeGlobal(int *hostIn) {
  int *deviceIn;
  hipMalloc((void **)&deviceIn, SIZE);
  hipMemcpy(deviceIn, hostIn, SIZE, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(Write, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, deviceIn);
  hipFree(deviceIn);
}


