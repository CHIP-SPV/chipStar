#include <hip/hip_runtime.h>
__global__ void k() {}
int main() {
  k<<<1, 1>>>();
  return hipDeviceSynchronize() != hipSuccess;
}
