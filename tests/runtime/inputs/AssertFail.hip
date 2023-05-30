// Regression test for https://github.com/CHIP-SPV/hipstar/issues/387.
#include <hip/hip_runtime.h>

__global__ void k() { __assert_fail("expression", "file", 123, "function"); }

int main() {
  k<<<1, 1>>>();
  (void)hipDeviceSynchronize();
  return 0;
}
