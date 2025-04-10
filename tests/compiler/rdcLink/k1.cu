#include "k.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

__global__ void saxpy3(int n, float a, float *x, float *y)
{
  int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  if (i < n) y[i] = a*x[i] + y[i];
}

__global__ void square_kernel(int* out, const int* in, const unsigned int size)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < size)
    {
        out[tid] = device_square(in[tid]);
    }
}

void test2(float *d_x,float *d_y, float *x, float *y, int N )
{
  hipLaunchKernelGGL(saxpy3,(N+255)/256, 256,0,0,N, 2.0f, d_x, d_y );
  hipDeviceSynchronize();
  float pABC[] = {1, 2, 3};
} 