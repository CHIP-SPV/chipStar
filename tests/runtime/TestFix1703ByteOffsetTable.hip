// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1703: &D[1] in a
// pointer table over an int array. Table entries are compared by address only.
#include <hip/hip_runtime.h>
#include <cstdio>

static const int D[4] = {10, 20, 30, 40};
__device__ const int *T[2] = {D, D + 1};

__global__ void k(size_t *O, unsigned I) {
  const int *L[2] = {D, D + 1};
  // T is filled by another kernel, which may see another copy of D (PoCL).
  O[0] = (size_t)T[I] - (size_t)T[0];
  O[1] = (size_t)L[I];
  O[2] = (size_t)&D[I];
  O[3] = D[I];
}

int main() {
  size_t *Dev, H[4] = {};
  hipError_t E = hipMalloc(&Dev, sizeof(H));
  if (E == hipSuccess) {
    k<<<1, 1>>>(Dev, 1);
    E = hipMemcpy(H, Dev, sizeof(H), hipMemcpyDeviceToHost);
  }
  bool Ok = E == hipSuccess && H[0] == sizeof(int) && H[1] == H[2] &&
            H[3] == 20;
  std::printf("err=%d file scope T[1] - T[0] %#zx, local %#zx, &D[1] %#zx, "
              "D[1] %zu %s\n",
              (int)E, H[0], H[1], H[2], H[3], Ok ? "PASS" : "FAIL");
  return !Ok;
}
