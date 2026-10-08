// Reproduces #1898: == and != on rank-1 HIP vectors failed to compile.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void k(bool *o, int1 a, int1 b, float1 c, float1 d) {
  o[0] = a == b;
  o[1] = a != b;
  o[2] = c == d;
  o[3] = c != d;
}

int main() {
  int1 a{1}, b{1};
  float1 c{1.0f}, d{2.0f};
  bool h[4], *dOut;
  if (hipMalloc(&dOut, sizeof(h)) != hipSuccess)
    return 1;
  hipLaunchKernelGGL(k, dim3(1), dim3(1), 0, 0, dOut, a, b, c, d);
  if (hipMemcpy(h, dOut, sizeof(h), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  hipFree(dOut);
  bool ok = h[0] && !h[1] && !h[2] && h[3] && (a == b) && !(a != b) &&
            !(c == d) && (c != d);
  printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
