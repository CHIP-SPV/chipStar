/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 * Copyright (c) 2022 Pekka Jääskeläinen / Parmance for Argonne National Laboratory
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
 * Test cases for abort() as specified by HIP:
 *
 *    "HIP does support an “abort” call which will terminate the process
 *     execution from inside the kernel."
 * https://rocmdocs.amd.com/en/latest/Programming_Guides/Kernel_language.html
 */

#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "hip/hip_runtime.h"

__global__ void cond_abort(int *Out) {
  if (Out[0] == 1) {
    printf("Aborting in kernel func...\n");
    abort();
    printf("Device side abort() survival detected! FAILED!\n");
  }
}

__attribute__((noinline)) __device__ int subfunc(int *Out) {
  if (Out[0] == 1) {
    printf("Aborting in subfunc...\n");
    abort();
    printf("Device side abort() survival detected in subfunc! FAILED!\n");
  }
  return 1;
}

__global__ void abort_in_subfunction(int *Out) {
  printf("Calling an aborting subfunc...\n");
  subfunc(Out);
  printf(
      "Aborting subfunction call survival detected in the caller! FAILED!\n");
}

int main(int argc, char *argv[]) {

  // Allow to test the device side control unwind behavior only without
  // aborting the process (unless the env is set already).
  setenv("CHIP_HOST_IGNORES_DEVICE_ABORT", "1", 1);

  uint NumThreads = 1;

  void *RetValVoid;
  hipHostMalloc(&RetValVoid, 4 * NumThreads);
  auto RetVal = reinterpret_cast<int *>(RetValVoid);

  RetVal[0] = 0;
  hipLaunchKernelGGL(cond_abort, dim3(1), dim3(1), 0, 0, RetVal);
  hipStreamSynchronize(0);

  RetVal[0] = 1;
  hipLaunchKernelGGL(cond_abort, dim3(1), dim3(1), 0, 0, RetVal);
  hipStreamSynchronize(0);

  hipLaunchKernelGGL(abort_in_subfunction, dim3(1), dim3(1), 0, 0, RetVal);
  hipStreamSynchronize(0);

  return 0;
}
