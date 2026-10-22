#include <cstdio>
#include <cuda_runtime.h>
__global__ void add(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() {
  const int n = 1024;
  float a[n], b[n], c[n];
  for (int i = 0; i < n; i++) { a[i] = i; b[i] = 2 * i; c[i] = 0; }
  float *da, *db, *dc;
  cudaMalloc(&da, n * sizeof(float));
  cudaMalloc(&db, n * sizeof(float));
  cudaMalloc(&dc, n * sizeof(float));
  cudaMemcpy(da, a, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice);
  add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
  cudaMemcpy(c, dc, n * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
    if (c[i] != 3.0f * i) { printf("FAILED at %d: %f\n", i, c[i]); return 1; }
  printf("PASSED\n");
  return 0;
}
