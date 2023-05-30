// Regression test for https://github.com/CHIP-SPV/hipstar/issues/387.
#include <hip/hip_runtime.h>
#include <cassert>

__global__ void k() { assert(false && "Hello, World!"); }

int main() {
  k<<<1, 1>>>();
  (void)hipDeviceSynchronize();
  return 0;
}
