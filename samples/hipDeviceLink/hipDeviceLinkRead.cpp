  #include "hipDeviceLink.h"

__device__ int global[NUM];

__global__ void Read(int *out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  out[tid] = global[tid];
}

/*
__global__ void Write(const int *in) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  global[tid] = in[tid];
}
*/

void readGlobal(int *hostOut) {
  int *deviceOut;
  hipMalloc((void **)&deviceOut, SIZE);
  hipLaunchKernelGGL(Read, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, deviceOut);
  hipMemcpy(hostOut, deviceOut, SIZE, hipMemcpyDeviceToHost);
  hipFree(deviceOut);
}

/*
void writeGlobal(int *hostIn) {
  int *deviceIn;
  hipMalloc((void **)&deviceIn, SIZE);
  hipMemcpy(deviceIn, hostIn, SIZE, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(Write, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, deviceIn);
  hipFree(deviceIn);
}
*/
