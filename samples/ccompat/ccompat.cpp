#include <stdio.h>
#include <hip/hip_runtime.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  hipMalloc(&d_x, N*sizeof(float)); 
  hipMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  hipMemcpy(d_x, x, N*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_y, y, N*sizeof(float), hipMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  hipMemcpy(y, d_y, N*sizeof(float), hipMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = std::max(maxError, std::abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  hipFree(d_x);
  hipFree(d_y);
  free(x);
  free(y);
}