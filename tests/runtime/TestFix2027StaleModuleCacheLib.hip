// Library half of TestFix2027StaleModuleCache; built twice with different VAL.
#include <hip/hip_runtime.h>

__device__ int G = VAL;

__global__ void k(int *O) { *O = VAL; }

extern "C" int run() {
  int *D, H = 0, V = 0;
  if (hipMalloc(&D, sizeof(int)) != hipSuccess)
    return -1;
  k<<<1, 1>>>(D);
  if (hipMemcpy(&H, D, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
    H = -2;
  (void)hipFree(D);
  if (hipMemcpyFromSymbol(&V, HIP_SYMBOL(G), sizeof(int)) != hipSuccess ||
      V != VAL)
    return -4;
  return H;
}
