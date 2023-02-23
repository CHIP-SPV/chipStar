/*
 * Copyright (c) 2022-2023 Pekka Jääskeläinen / Intel Finland Oy
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

template <typename T>
__global__ void shfl_down(T *In, T *Out, int Delta, int Width) {
  Out[threadIdx.x] = __shfl_down(In[threadIdx.x], Delta, Width);
}

template <typename T>
__global__ void shfl_up(T *In, T *Out, int Delta, int Width) {
  Out[threadIdx.x] = __shfl_up(In[threadIdx.x], Delta, Width);
}

template <typename T> __global__ void shfl(T *In, T *Out, int Lane, int Width) {
  Out[threadIdx.x] = __shfl(In[threadIdx.x], Lane, Width);
}

template <typename T>
__global__ void shfl_xor(T *In, T *Out, int Mask, int Width) {
  Out[threadIdx.x] = __shfl_xor(In[threadIdx.x], Mask, Width);
}

__global__ void test_lane_id(unsigned *Out) { Out[threadIdx.x] = __lane_id(); }

#define LAUNCH_CASE_T(TYPE, KERNEL, DELTA, WIDTH)                              \
  std::iota((TYPE *)Input, (TYPE *)Input + Threads, (TYPE)0);                  \
  std::fill((TYPE *)Output, (TYPE *)Output + Threads, (TYPE)0);                \
                                                                               \
  hipLaunchKernelGGL(KERNEL<TYPE>, dim3(CHIP_DEFAULT_WARP_SIZE),               \
                     dim3(Threads), 0, 0, (TYPE *)Input, (TYPE *)Output,       \
                     DELTA, WIDTH);                                            \
  (void)hipStreamSynchronize(0);                                               \
                                                                               \
  std::cout << #KERNEL "<" #TYPE << "> arg=" << DELTA << " width=" << WIDTH    \
            << ": " << std::endl;                                              \
  for (int i = 0; i < Threads;) {                                              \
    for (int wi = 0; wi < WIDTH; ++wi, ++i)                                    \
      std::cout << ((TYPE *)Output)[i] << ' ';                                 \
    std::cout << std::endl;                                                    \
  }                                                                            \
  std::cout << std::endl

#define LAUNCH_CASE(KERNEL, DELTA, WIDTH)                                      \
  LAUNCH_CASE_T(int, KERNEL, DELTA, WIDTH);                                    \
  LAUNCH_CASE_T(unsigned int, KERNEL, DELTA, WIDTH);                           \
  LAUNCH_CASE_T(long, KERNEL, DELTA, WIDTH);                                   \
  LAUNCH_CASE_T(unsigned long, KERNEL, DELTA, WIDTH);                          \
  LAUNCH_CASE_T(long long, KERNEL, DELTA, WIDTH);                              \
  LAUNCH_CASE_T(unsigned long long, KERNEL, DELTA, WIDTH);                     \
  LAUNCH_CASE_T(float, KERNEL, DELTA, WIDTH);                                  \
  LAUNCH_CASE_T(double, KERNEL, DELTA, WIDTH);

int main(int argc, char *argv[]) {

  constexpr uint Threads = CHIP_DEFAULT_WARP_SIZE * 2;
  constexpr unsigned MaxEltSize = sizeof(double);

  void *OutputVoid;
  hipHostMalloc(&OutputVoid, MaxEltSize * Threads);
  auto Output = reinterpret_cast<int *>(OutputVoid);

  void *InputVoid;
  hipHostMalloc(&InputVoid, MaxEltSize * Threads);
  auto Input = reinterpret_cast<int *>(InputVoid);

  LAUNCH_CASE(shfl_down, 4, CHIP_DEFAULT_WARP_SIZE);
  LAUNCH_CASE(shfl_up, 4, CHIP_DEFAULT_WARP_SIZE);

  LAUNCH_CASE(shfl_down, 4, CHIP_DEFAULT_WARP_SIZE / 2);
  LAUNCH_CASE(shfl_up, 4, CHIP_DEFAULT_WARP_SIZE / 2);

  LAUNCH_CASE(shfl_down, 1, 4);
  LAUNCH_CASE(shfl_up, 1, 4);

  LAUNCH_CASE(shfl, 15, 16);
  LAUNCH_CASE(shfl, 0, 16);

  LAUNCH_CASE(shfl_xor, 3, CHIP_DEFAULT_WARP_SIZE);
  LAUNCH_CASE(shfl_xor, 5, CHIP_DEFAULT_WARP_SIZE);

  std::fill(Output, Output + Threads, 0);

  hipLaunchKernelGGL(test_lane_id, dim3(CHIP_DEFAULT_WARP_SIZE), dim3(Threads),
                     0, 0, (unsigned int *)Output);
  hipStreamSynchronize(0);

  std::cout << "lane_id_test: " << std::endl;
  for (int i = 0; i < Threads;) {
    for (int wi = 0; wi < Threads; ++wi, ++i)
      std::cout << Output[i] << ' ';
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
