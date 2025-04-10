#include "k.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

__global__ void saxpy2(int n, float a, float *x, float *y)
{
  int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  if (i < n) y[i] = a*x[i] + y[i];
}

__device__ int device_square(int x)
{
    return x * x;
}

void test(float *d_x,float *d_y, float *x, float *y, int N )
{
  hipLaunchKernelGGL(saxpy2,(N+255)/256, 256,0,0,N, 2.0f, d_x, d_y );
  hipDeviceSynchronize();
} 