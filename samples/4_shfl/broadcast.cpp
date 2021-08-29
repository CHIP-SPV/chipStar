#include "hip/hip_runtime.h"

#include <iostream>

#define BUF_SIZE 256
#define WARP_MASK 0x7
#define WARP_SUM 28

#define HIPCHECK(code)                                                         \
  do {                                                                         \
    hiperr = code;                                                             \
    if (hiperr != hipSuccess) {                                                \
      std::cerr << "ERROR on line " << __LINE__ << ": " << (unsigned)hiperr    \
                << "\n";                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void bcast(int *out) {
  int value = (hipThreadIdx_x & WARP_MASK);

  for (int mask = 1; mask < WARP_MASK; mask *= 2)
    value += __shfl_xor(value, mask);

  size_t oi = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

  out[oi] = value;
}

int main() {

  int *out = (int *)malloc(sizeof(int) * BUF_SIZE);
  int *d_out;
  hipError_t hiperr = hipSuccess;

  HIPCHECK(hipMalloc((void **)&d_out, sizeof(int) * BUF_SIZE));

  hipLaunchKernelGGL(bcast, dim3(1), dim3(BUF_SIZE), 0, 0, d_out);
  HIPCHECK(hipGetLastError());

  HIPCHECK(
      hipMemcpy(out, d_out, sizeof(int) * BUF_SIZE, hipMemcpyDeviceToHost));

  size_t errs = 0;
  for (int i = 0; i < BUF_SIZE; i++) {
    if (out[i] != WARP_SUM) {
      std::cout << "ERROR @ " << i << ":  " << out[i] << "\n";
      ++errs;
    }
  }

  free(out);
  HIPCHECK(hipFree(d_out));

  if (errs != 0) {
    std::cout << "FAILED: " << errs << " errors\n";
    return 1;
  } else {
    std::cout << "PASSED!\n";
    return 0;
  }
}
