/*
 * Copyright (c) 2021-22 chipStar developers
 * Copyright (c) 2020 Michal Babej <michal.babej@tuni.fi>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#include "hip/hip_runtime.h"

#include <iostream>

#define BUF_SIZE 256
#define WARP_MASK 0x7
#define EXPECTED 12345

#define HIPCHECK(code)                                                         \
  do {                                                                         \
    hiperr = code;                                                             \
    if (hiperr != hipSuccess) {                                                \
      std::cerr << "ERROR on line " << __LINE__ << ": " << (unsigned)hiperr    \
                << "\n";                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void bcast(int arg, int *out) {
  int value = ((hipThreadIdx_x & WARP_MASK) == 0) ? arg : 0;

  int out_v = __shfl(
      value, 0); // Synchronize all threads in warp, and get "value" from lane 0

  size_t oi = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  out[oi] = out_v;
}

int main() {

  int *out = (int *)malloc(sizeof(int) * BUF_SIZE);
  int *d_out;
  hipError_t hiperr = hipSuccess;

  HIPCHECK(hipMalloc((void **)&d_out, sizeof(int) * BUF_SIZE));

  hipLaunchKernelGGL(bcast, dim3(1), dim3(BUF_SIZE), 0, 0, EXPECTED, d_out);
  HIPCHECK(hipGetLastError());

  HIPCHECK(
      hipMemcpy(out, d_out, sizeof(int) * BUF_SIZE, hipMemcpyDeviceToHost));

  size_t errs = 0;
  for (int i = 0; i < BUF_SIZE; i++) {
    if (out[i] != EXPECTED) {
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
