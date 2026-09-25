// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1691: a local
// array initialized with __shared__ addresses must hold those addresses.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void k(size_t *Out, unsigned C) {
  __shared__ float A[64], B[64];
  float *Buf[2] = {A, B};
  // Optimized, the part copied to Q is read straight from the initializer.
  float *P[4] = {B, A, B, A}, *Q[2];
  __builtin_memcpy(Q, P + 1, sizeof(Q));
  Out[0] = (size_t)((C & 1) ? B : A);
  Out[1] = (size_t)Buf[C & 1];
  Out[2] = (size_t)Q[C & 1];
}

int main() {
  size_t *D, H[3] = {};
  bool Ok = hipMalloc(&D, sizeof(H)) == hipSuccess;
  for (unsigned C = 0; Ok && C < 2; ++C) {
    hipError_t Z = hipMemset(D, 0xff, sizeof(H));
    k<<<1, 1>>>(D, C);
    hipError_t E = hipMemcpy(H, D, sizeof(H), hipMemcpyDeviceToHost);
    // All-ones is never an address, so a launch that did not run cannot pass.
    Ok = Z == hipSuccess && E == hipSuccess && H[0] != SIZE_MAX &&
         H[1] == H[0] && H[2] == H[0];
    std::printf("C=%u err=%d direct=0x%zx table=0x%zx copy=0x%zx\n", C, (int)E,
                H[0], H[1], H[2]);
  }
  std::printf("%s\n", Ok ? "PASS" : "FAIL");
  return !Ok;
}
