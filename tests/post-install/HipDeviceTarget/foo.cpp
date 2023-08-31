#include <hip/hip_runtime.h>
__global__ void k() { printf("Hello, World!\n"); }
int main() {
  k<<<1, 1>>>();
  bool error = (hipGetLastError() != hipSuccess) ||
               (hipDeviceSynchronize() != hipSuccess);
  return error;
}
