// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1693 and
// https://github.com/CHIP-SPV/chipStar/issues/1695: "abc" + 1 in a string
// table. Only addresses are compared: dereferencing a bad one can lose the
// device.
#include <hip/hip_runtime.h>
#include <cstdio>

__device__ const char *Names[3] = {"abc", "def", "abc" + 1};

__global__ void k(size_t *O, unsigned C) {
  const char *Local[3] = {"abc", "def", "abc" + 1};
  O[0] = (size_t)Names[C];
  O[1] = (size_t)Local[C];
}

int main() {
  size_t *D, H[2][2] = {};
  hipError_t E = hipMalloc(&D, sizeof(H[0]));
  for (unsigned I = 0; E == hipSuccess && I < 2; ++I) {
    k<<<1, 1>>>(D, 2 * I);
    E = hipMemcpy(H[I], D, sizeof(H[I]), hipMemcpyDeviceToHost);
  }
  bool Ok = E == hipSuccess && H[1][0] == H[0][0] + 1 && H[1][1] == H[0][1] + 1;
  std::printf("err=%d file scope %#zx %#zx, local %#zx %#zx %s\n", (int)E,
              H[0][0], H[1][0], H[0][1], H[1][1], Ok ? "PASS" : "FAIL");
  return !Ok;
}
