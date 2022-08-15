/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 * Copyright (c) 2022 Henry Linjamäki / Parmance for Argonne National Laboratory
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
// Check global device variables are reset to their initialization
// value after hipDeviceReset() call.

#include <hip/hip_runtime.h>
#include "test_common.h"

__device__ int A = 123;
__constant__ int B = 321;

__global__ void ReadAB(int *Out) {
  Out[0] = A;
  Out[1] = B;
}

int main() {
  int AData = 1;
  int BData = 2;
  HIPCHECK(hipMemcpyToSymbol(A, &AData, sizeof(int)));
  HIPCHECK(hipMemcpyToSymbol(B, &BData, sizeof(int)));

  int *OutD;
  HIPCHECK(hipMalloc(&OutD, 2 * sizeof(int)));
  hipLaunchKernelGGL(ReadAB, dim3(1), dim3(1), 0, 0, OutD);

  int OutH[2];
  HIPCHECK(hipMemcpy(&OutH, OutD, 2 * sizeof(int), hipMemcpyDeviceToHost));
  HIPASSERT(OutH[0] == 1);
  HIPASSERT(OutH[1] == 2);

  HIPCHECK(hipDeviceReset());

  HIPCHECK(hipMemcpyFromSymbol(&OutH[0], A, sizeof(int)));
  HIPCHECK(hipMemcpyFromSymbol(&OutH[1], B, sizeof(int)));
  HIPASSERT(OutH[0] == 123);
  HIPASSERT(OutH[1] == 321);

  HIPCHECK(hipFree(OutD));
  passed();
}
