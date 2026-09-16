// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1641: a device
// variable named `all`.
#include <hip/hip_runtime.h>
#include <cstdio>

__device__ int all = 5;

__global__ void k(int *O) { O[0] = all; }

int main() {
  int *D, H = -1;
  (void)hipMalloc(&D, sizeof(H));
  k<<<1, 1>>>(D);
  hipError_t E = hipMemcpy(&H, D, sizeof(H), hipMemcpyDeviceToHost);
  bool Ok = E == hipSuccess && H == 5;
  std::printf("err=%d all=%d %s\n", (int)E, H, Ok ? "PASS" : "FAIL");
  return !Ok;
}
