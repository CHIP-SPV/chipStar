#include <hip/hip_runtime.h>

__global__ void kernel(float* x, float* y, int n) {
  size_t tid{threadIdx.x};
  if (tid < 1) {
    for (int i = 0; i < n; i++) {
      x[i] = sqrt(powf(3.14159, i));
    }
    y[tid] = y[tid] + 1.0f;
  }
}

int main() {
  float* x;
  float* y;
  hipMalloc(&x, 1024 * sizeof(float));
  hipMalloc(&y, 1024 * sizeof(float));
  hipLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, 0, x, y, 1024);
  hipDeviceSynchronize();
  return 0;

}