/*
 * Copyright (c) 2022 Pekka Jääskeläinen / Intel
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

/*
 * Test cases for the shuffle variations.
 */

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <numeric>

#include "hip/hip_runtime.h"

__global__ void shfl_down_int(int *In, int *Out, int Delta, int Width) {
  Out[threadIdx.x] = __shfl_down(In[threadIdx.x], Delta, Width);
}

__global__ void shfl_up_int(int *In, int *Out, int Delta, int Width) {
  Out[threadIdx.x] = __shfl_up(In[threadIdx.x], Delta, Width);
}

__global__ void shfl_int(int *In, int *Out, int Lane, int Width) {
  Out[threadIdx.x] = __shfl(In[threadIdx.x], Lane, Width);
}

__global__ void shfl_down_float(float *In, float *Out, int Delta, int Width) {
  Out[threadIdx.x] = __shfl_down(In[threadIdx.x], Delta, Width);
}

__global__ void shfl_up_float(float *In, float *Out, int Delta, int Width) {
  Out[threadIdx.x] = __shfl_up(In[threadIdx.x], Delta, Width);
}

__global__ void shfl_float(float *In, float *Out, int Lane, int Width) {
  Out[threadIdx.x] = __shfl(In[threadIdx.x], Lane, Width);
}

#define LAUNCH_CASE_T(TYPE, KERNEL, DELTA, WIDTH)                              \
  std::iota(Input, Input + Threads, (TYPE)0);                                  \
  std::fill(Output, Output + Threads, (TYPE)0);                                \
                                                                               \
  hipLaunchKernelGGL(KERNEL##_##TYPE, dim3(CHIP_DEFAULT_WARP_SIZE),            \
                     dim3(Threads), 0, 0, (TYPE *)Input, (TYPE *)Output,       \
                     DELTA, WIDTH);                                            \
  hipStreamSynchronize(0);                                                     \
                                                                               \
  std::cout << #KERNEL "_" #TYPE << " arg=" << DELTA << " width=" << WIDTH     \
            << ": " << std::endl;                                              \
  for (int i = 0; i < Threads;) {                                              \
    for (int wi = 0; wi < WIDTH; ++wi, ++i)                                    \
      std::cout << (TYPE)Output[i] << ' ';                                     \
    std::cout << std::endl;                                                    \
  }                                                                            \
  std::cout << std::endl

#define LAUNCH_CASE(KERNEL, DELTA, WIDTH)                                      \
  LAUNCH_CASE_T(int, KERNEL, DELTA, WIDTH);                                    \
  LAUNCH_CASE_T(float, KERNEL, DELTA, WIDTH)

int main(int argc, char *argv[]) {

  uint Threads = CHIP_DEFAULT_WARP_SIZE * 2;

  void *OutputVoid;
  hipHostMalloc(&OutputVoid, sizeof(int) * Threads);
  auto Output = reinterpret_cast<int *>(OutputVoid);

  void *InputVoid;
  hipHostMalloc(&InputVoid, sizeof(int) * Threads);
  auto Input = reinterpret_cast<int *>(InputVoid);

  LAUNCH_CASE(shfl_down, 4, CHIP_DEFAULT_WARP_SIZE);
  LAUNCH_CASE(shfl_up, 4, CHIP_DEFAULT_WARP_SIZE);

  LAUNCH_CASE(shfl_down, 4, CHIP_DEFAULT_WARP_SIZE / 2);
  LAUNCH_CASE(shfl_up, 4, CHIP_DEFAULT_WARP_SIZE / 2);

  LAUNCH_CASE(shfl_down, 1, 4);
  LAUNCH_CASE(shfl_up, 1, 4);

  LAUNCH_CASE(shfl, 15, 16);
  LAUNCH_CASE(shfl, 0, 16);

  return 0;
}
