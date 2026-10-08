// Reproduces #1903: == and != on HIP vectors emitted recursive device code.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void k(bool *o, int2 a, int2 b, int3 c, int3 d, float4 e, float4 f) {
  o[0] = a == b;
  o[1] = a != b;
  o[2] = c == c;
  o[3] = c != d;
  o[4] = e == f;
  o[5] = e != f;
}

int main() {
  int2 a{1, 2}, b{1, 2};
  int3 c{1, 2, 3}, d{0, 2, 3};
  float4 e{1, 2, 3, 4}, f{0, 2, 3, 4};
  bool h[6] = {}, *dOut;
  if (hipMalloc(&dOut, sizeof(h)) != hipSuccess)
    return 1;
  hipLaunchKernelGGL(k, dim3(1), dim3(1), 0, 0, dOut, a, b, c, d, e, f);
  if (hipGetLastError() != hipSuccess)
    return 1;
  if (hipMemcpy(h, dOut, sizeof(h), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  hipFree(dOut);
  bool ok = h[0] && !h[1] && h[2] && h[3] && !h[4] && h[5];
  printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
