/*
 * Copyright (c) 2022 chipStar developers
 * Copyright (c) 2022 Paulius Velesko <pvelesko@pglc.io>
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

#include <cassert>
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
  int numBlocks = 5120000;
  int dimBlocks = 1;
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
  int count = 10000;

  CHECK(hipEventCreate(&start));
  CHECK(hipEventCreate(&stop));
  assert(hipEventQuery(stop) == hipErrorNotReady);
  assert(hipEventQuery(start) == hipErrorNotReady);

  CHECK(hipEventRecord(start));

  std::cout << "Launching kernel\n";
  hipLaunchKernelGGL(addCountReverse, dim3(numBlocks), dim3(dimBlocks),
                     sharedMem, 0, A_d, C_d, NUM, count);
  CHECK(hipGetLastError());
  std::cout << "Kernel launched successfully\n";

  CHECK(hipEventRecord(stop));

  assert(hipEventQuery(stop) == hipErrorNotReady);

  float t;
  hipError_t notReady = hipEventElapsedTime(&t, start, stop);
  std::cout << "Kernel time: " << t << "s\n";

  if (notReady == hipErrorNotReady) {
    std::cout << "PASSED!" << std::endl;
  } else {
    std::cout << "FAILED!" << std::endl;
  }

  // Can't guarantee the test completes within test time limit. The
  // kernel may take very long time and backends may have to wait its
  // completion before they can release their resources.
  std::quick_exit(!(notReady == hipErrorNotReady));
}
