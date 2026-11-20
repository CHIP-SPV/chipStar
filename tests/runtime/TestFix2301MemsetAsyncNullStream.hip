// Reproduces https://github.com/CHIP-SPV/chipStar/issues/2301: hipMemset2DAsync/hipMemset3DAsync reject a null stream.
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  const size_t W = 16, H = 8, D = 4, N = W * H * D;
  unsigned char *P, Host[N];
  if (hipMalloc(&P, N) != hipSuccess)
    return 1;

  hipError_t E = hipMemset2DAsync(P, W, 0x5a, W, H, nullptr);
  if (E != hipSuccess) {
    printf("FAIL: hipMemset2DAsync(nullptr stream) returned %d\n", (int)E);
    return 1;
  }
  hipStreamSynchronize(nullptr);
  hipMemcpy(Host, P, W * H, hipMemcpyDeviceToHost);
  for (size_t I = 0; I < W * H; ++I)
    if (Host[I] != 0x5a) {
      printf("FAIL: 2D byte %zu is 0x%x\n", I, Host[I]);
      return 1;
    }

  E = hipMemset3DAsync(hipPitchedPtr{P, W, W, H}, 0x3c, hipExtent{W, H, D},
                       nullptr);
  if (E != hipSuccess) {
    printf("FAIL: hipMemset3DAsync(nullptr stream) returned %d\n", (int)E);
    return 1;
  }
  hipStreamSynchronize(nullptr);
  hipMemcpy(Host, P, N, hipMemcpyDeviceToHost);
  for (size_t I = 0; I < N; ++I)
    if (Host[I] != 0x3c) {
      printf("FAIL: 3D byte %zu is 0x%x\n", I, Host[I]);
      return 1;
    }
  printf("PASS\n");
  return 0;
}
