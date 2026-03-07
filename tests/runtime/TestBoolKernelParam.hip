#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void boolKernel(bool flag, int* out) {
  if (flag)
    *out = 42;
  else
    *out = -1;
}

__global__ void multiBoolKernel(bool a, int x, bool b, int* out) {
  *out = (a ? 1 : 0) + x + (b ? 100 : 0);
}

int main() {
  int *d_out;
  int h_out;
  hipMalloc(&d_out, sizeof(int));

  // Test 1: bool = true
  h_out = 0;
  hipMemcpy(d_out, &h_out, sizeof(int), hipMemcpyHostToDevice);
  boolKernel<<<1, 1>>>(true, d_out);
  hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost);
  if (h_out != 42) {
    printf("FAIL: boolKernel(true) expected 42, got %d\n", h_out);
    return 1;
  }

  // Test 2: bool = false
  h_out = 0;
  hipMemcpy(d_out, &h_out, sizeof(int), hipMemcpyHostToDevice);
  boolKernel<<<1, 1>>>(false, d_out);
  hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost);
  if (h_out != -1) {
    printf("FAIL: boolKernel(false) expected -1, got %d\n", h_out);
    return 1;
  }

  // Test 3: mixed bool and int params
  h_out = 0;
  hipMemcpy(d_out, &h_out, sizeof(int), hipMemcpyHostToDevice);
  multiBoolKernel<<<1, 1>>>(true, 10, true, d_out);
  hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost);
  if (h_out != 111) {
    printf("FAIL: multiBoolKernel(true, 10, true) expected 111, got %d\n", h_out);
    return 1;
  }

  printf("PASS\n");
  hipFree(d_out);
  return 0;
}
