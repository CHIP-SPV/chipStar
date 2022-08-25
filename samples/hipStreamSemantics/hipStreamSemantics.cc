/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 * Copyright (c) 2021-22 Paulius Velesko <pvelesko@pglc.io>
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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define CHECK(cmd)                                                             \
  {                                                                            \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),  \
              error, __FILE__, __LINE__);                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

__global__ void addOne(int *__restrict A) {
  const uint i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  A[i] = A[i] + 1;
}

template <typename T>
__global__ void addCountReverse(const T *A_d, T *C_d, int64_t NELEM,
                                int count) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  // Deliberately do this in an inefficient way to increase kernel runtime
  for (int i = 0; i < count; i++) {
    for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
      C_d[i] = A_d[i] + (T)count;
    }
  }
}

int main() {
  int numBlocks = 512000;
  int dimBlocks = 32;
  const size_t NUM = numBlocks * dimBlocks;
  int *A_h, *A_d, *Ref;
  int *C_h, *C_d;

  static int device = 0;
  CHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
  printf("info: running on device %s\n", props.name);

  A_h = (int *)calloc(NUM, sizeof(int));
  C_h = (int *)calloc(NUM, sizeof(int));
  CHECK(hipMalloc((void **)&A_d, NUM * sizeof(int)));
  CHECK(hipMalloc((void **)&C_d, NUM * sizeof(int)));
  printf("info: copy Host2Device\n");
  CHECK(hipMemcpy(A_d, A_h, NUM * sizeof(int), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(C_d, C_h, NUM * sizeof(int), hipMemcpyHostToDevice));

  hipStream_t q;
  uint32_t flags = hipStreamNonBlocking;
  CHECK(hipStreamCreateWithFlags(&q, flags));

  size_t sharedMem = 0;
  hipEvent_t start, stop;
  int count = 1000;

  CHECK(hipEventCreate(&start));
  CHECK(hipEventCreate(&stop));
  CHECK(hipEventRecord(start));
  std::cout << "Launching kernel\n";
  hipLaunchKernelGGL(addCountReverse, dim3(numBlocks), dim3(dimBlocks),
                     sharedMem, 0, A_d, C_d, NUM, count);
  CHECK(hipEventRecord(stop));
  std::cout << "Kernel submitted to queue\n";
  CHECK(hipGetLastError());
  float t;
  CHECK(hipDeviceSynchronize());
  CHECK(hipEventElapsedTime(&t, start, stop));
  std::cout << "Kernel time: " << t << "s\n";

  hipLaunchKernelGGL(addOne, dim3(numBlocks), dim3(dimBlocks), sharedMem, q,
                     A_d);
  hipLaunchKernelGGL(addOne, dim3(numBlocks), dim3(dimBlocks), sharedMem, q,
                     A_d);
  hipLaunchKernelGGL(addOne, dim3(numBlocks), dim3(dimBlocks), sharedMem, q,
                     A_d);
  hipLaunchKernelGGL(addOne, dim3(numBlocks), dim3(dimBlocks), sharedMem, 0,
                     A_d);
  hipMemcpy(A_h, A_d, NUM * sizeof(int), hipMemcpyDeviceToHost);

  bool pass = true;
  int num_errors = 0;
  for (int i = 0; i < NUM; i++) {
    if (A_h[i] != 4) {
      pass = false;
      num_errors++;
    }
  }

  std::cout << "Num Errors: " << num_errors << std::endl;
  std::cout << (pass ? "PASSED!" : "FAIL") << std::endl;
}
