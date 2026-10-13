// Device warpSize must be the width of the subgroup the kernel runs at.
// The indirect call keeps HipWarpsPass from pinning the kernel to 32, so under
// IGC_ForceOCLSIMDWidth=16 (set by CMake) IGC compiles it at SIMD16.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

static __device__ int one(int X) { return X + 1; }
static __device__ int two(int X) { return X + 2; }

__global__ void k(int *Out, int Sel) {
  int (*Fns[])(int) = {one, two};
  int L = threadIdx.x;
  if (L == 0)
    Out[64] = Fns[Sel](0);
  Out[L] = warpSize * 1000 + (int)__popcll(__ballot(1));
  // Default width: lanes in the upper half keep their own value.
  Out[65 + L] = __shfl_down(L, warpSize / 2) - L;
}

int main() {
  hipDeviceProp_t Props;
  (void)hipGetDeviceProperties(&Props, 0);
  // Only Intel drivers take SPV_INTEL_function_pointers. PoCL names its
  // device "cpu-<arch>-...Intel(R)...", so match the prefix.
  if (strncmp(Props.name, "Intel(R)", 8)) {
    printf("HIP_SKIP_THIS_TEST: no indirect calls on %s\n", Props.name);
    return CHIP_SKIP_TEST;
  }
  int *D = nullptr, H[129];
  (void)hipMalloc(&D, sizeof(H));
  k<<<1, 64>>>(D, 1);
  (void)hipMemcpy(H, D, sizeof(H), hipMemcpyDeviceToHost);
  (void)hipFree(D);

  int Bad = 0;
  if (H[64] != 2) {
    printf("FAIL: indirect call returned %d\n", H[64]);
    Bad++;
  }
  for (int L = 0; L < 64; L++) {
    int W = H[L] / 1000;
    if (W == 0 || W != H[L] % 1000) {
      printf("FAIL: lane %d warpSize %d, subgroup has %d lanes\n", L, W,
             H[L] % 1000);
      Bad++;
      continue;
    }
    int Want = L % W < W / 2 ? W / 2 : 0;
    if (H[65 + L] != Want) {
      printf("FAIL: lane %d __shfl_down moved %d lanes, want %d\n", L,
             H[65 + L], Want);
      Bad++;
    }
  }
  if (!Bad)
    printf("PASSED (warpSize %d)\n", H[0] / 1000);
  return Bad ? 1 : 0;
}
