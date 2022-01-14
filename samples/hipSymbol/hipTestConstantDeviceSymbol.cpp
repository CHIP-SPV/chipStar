#include <hip/hip_runtime.h>
#include "test_common.h"

__constant__ __device__ int ConstOut = 123;
__constant__ __device__ int ConstIn = 321;

__global__ void Assign(int* Out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0)
    Out[tid] = -ConstIn;
}

int main() {
  int Ah;
  hipMemcpyFromSymbol(&Ah, HIP_SYMBOL(ConstOut), sizeof(int));
  assert(Ah == 123);

  int Bh = 654, Ch, *Cd;
  hipMalloc((void**)&Cd, sizeof(int));
  hipMemcpyToSymbol(HIP_SYMBOL(ConstIn), &Bh, sizeof(int));
  hipLaunchKernelGGL(Assign, dim3(1), dim3(1), 0, 0, Cd);
  hipMemcpy(&Ch, Cd, sizeof(int), hipMemcpyDeviceToHost);
  assert(Ch == -654);

  hipFree(Cd);
  passed();
}
