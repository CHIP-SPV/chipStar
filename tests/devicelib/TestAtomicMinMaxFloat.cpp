#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

__global__ void testAtomicMaxFloat(float* out) {
  atomicMax(out, (float)threadIdx.x);
}

__global__ void testAtomicMinFloat(float* out) {
  atomicMin(out, (float)threadIdx.x);
}

__global__ void testAtomicMaxDouble(double* out) {
  atomicMax(out, (double)threadIdx.x);
}

__global__ void testAtomicMinDouble(double* out) {
  atomicMin(out, (double)threadIdx.x);
}

int main() {
  float *d_fmax, *d_fmin;
  double *d_dmax, *d_dmin;
  float h_fmax, h_fmin;
  double h_dmax, h_dmin;

  hipMalloc(&d_fmax, sizeof(float));
  hipMalloc(&d_fmin, sizeof(float));
  hipMalloc(&d_dmax, sizeof(double));
  hipMalloc(&d_dmin, sizeof(double));

  // Test atomicMax float
  h_fmax = 0.0f;
  hipMemcpy(d_fmax, &h_fmax, sizeof(float), hipMemcpyHostToDevice);
  testAtomicMaxFloat<<<1, 64>>>(d_fmax);
  hipMemcpy(&h_fmax, d_fmax, sizeof(float), hipMemcpyDeviceToHost);
  if (h_fmax != 63.0f) {
    printf("FAIL: atomicMax float expected 63.0, got %f\n", h_fmax);
    return 1;
  }

  // Test atomicMin float
  h_fmin = 100.0f;
  hipMemcpy(d_fmin, &h_fmin, sizeof(float), hipMemcpyHostToDevice);
  testAtomicMinFloat<<<1, 64>>>(d_fmin);
  hipMemcpy(&h_fmin, d_fmin, sizeof(float), hipMemcpyDeviceToHost);
  if (h_fmin != 0.0f) {
    printf("FAIL: atomicMin float expected 0.0, got %f\n", h_fmin);
    return 1;
  }

  // Test atomicMax double
  h_dmax = 0.0;
  hipMemcpy(d_dmax, &h_dmax, sizeof(double), hipMemcpyHostToDevice);
  testAtomicMaxDouble<<<1, 64>>>(d_dmax);
  hipMemcpy(&h_dmax, d_dmax, sizeof(double), hipMemcpyDeviceToHost);
  if (h_dmax != 63.0) {
    printf("FAIL: atomicMax double expected 63.0, got %f\n", h_dmax);
    return 1;
  }

  // Test atomicMin double
  h_dmin = 100.0;
  hipMemcpy(d_dmin, &h_dmin, sizeof(double), hipMemcpyHostToDevice);
  testAtomicMinDouble<<<1, 64>>>(d_dmin);
  hipMemcpy(&h_dmin, d_dmin, sizeof(double), hipMemcpyDeviceToHost);
  if (h_dmin != 0.0) {
    printf("FAIL: atomicMin double expected 0.0, got %f\n", h_dmin);
    return 1;
  }

  printf("PASS\n");

  hipFree(d_fmax);
  hipFree(d_fmin);
  hipFree(d_dmax);
  hipFree(d_dmin);
  return 0;
}
