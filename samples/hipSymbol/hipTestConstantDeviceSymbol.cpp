/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 * Copyright (c) 2022 Henry Linjamäki / Parmance for Argonne National
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

#include <hip/hip_runtime.h>
#include "test_common.h"

__constant__ __device__ int ConstOut = 123;
__constant__ __device__ int ConstIn = 321;

__global__ void Assign(int* Out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0)
    Out[tid] = -ConstIn;
}

int main() {
  int Ah;
  hipMemcpyFromSymbol(&Ah, HIP_SYMBOL(ConstOut), sizeof(int));
  assert(Ah == 123);

  int Bh = 654, Ch, *Cd;
  hipMalloc((void**)&Cd, sizeof(int));
  hipMemcpyToSymbol(HIP_SYMBOL(ConstIn), &Bh, sizeof(int));
  hipLaunchKernelGGL(Assign, dim3(1), dim3(1), 0, 0, Cd);
  hipMemcpy(&Ch, Cd, sizeof(int), hipMemcpyDeviceToHost);
  assert(Ch == -654);

  hipFree(Cd);
  passed();
}
