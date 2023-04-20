#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void kernel() { printf("Hello World!"); }

int main() {
  kernel<<<1, 1>>>();
  hipDeviceSynchronize();
  return 0;
}
