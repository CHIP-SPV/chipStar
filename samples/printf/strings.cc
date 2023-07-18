/*
 * Copyright (c) 2021-22 chipStar developers
 * Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved
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
 * Test cases for %s which are special cases due to OpenCL requiring their
 * arguments to be literals. Some code adopted from hipPrintfBasic.cpp of
 * the HIP test suite. Run only one work-item at a time since otherwise
 * the printout (order) is undefined and difficult to check.
 */

#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "hip/hip_runtime.h"
#include "printf_tests_common.h"

__global__ void no_args(int *out) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  printf("## no_args\n");
  out[tid] = printf("Hello\n");
}

__global__ void literal_str_arg(int *out) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  printf("## literal_str_arg\n");
  out[tid] = printf("%s", "Hello");
  out[tid] += printf(" %s\n", "strings.");
}

__global__ void var_str_arg(int *out) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  const char *hello = "HELLO\n";
  const char *strings = "STRiNGS\n";
  printf("## global_str_arg\n");
  out[tid] = printf("%s", hello);
  out[tid] += printf(" %s\n", strings);
}

int main(int argc, char *argv[]) {
  uint num_threads = 1;
  uint failures = 0;

  void *retval_void;
  CHECK(hipHostMalloc(&retval_void, 4 * num_threads), hipSuccess);
  auto retval = reinterpret_cast<int *>(retval_void);

  hipLaunchKernelGGL(no_args, dim3(1), dim3(1), 0, 0, retval);
  hipStreamSynchronize(0);

  for (uint ii = 0; ii != num_threads; ++ii) {
#ifdef __HIP_PLATFORM_AMD__
    CHECK(retval[ii], strlen("Hello there.\n"));
#else
    CHECK(retval[ii], 0);
#endif
  }

  hipLaunchKernelGGL(literal_str_arg, dim3(1), dim3(1), 0, 0, retval);
  hipStreamSynchronize(0);

  for (uint ii = 0; ii != num_threads; ++ii) {
#ifdef __HIP_PLATFORM_AMD__
    CHECK(retval[ii], 0);
#else
    CHECK(retval[ii], 2);
#endif
  }

  hipLaunchKernelGGL(var_str_arg, dim3(1), dim3(1), 0, 0, retval);
  hipStreamSynchronize(0);

  for (uint ii = 0; ii != num_threads; ++ii) {
#ifdef __HIP_PLATFORM_AMD__
    CHECK(retval[ii], 0);
#else
    CHECK(retval[ii], 2);
#endif
  }
  printf((failures == 0) ? "PASSED!\n" : "FAILED!\n");
}
