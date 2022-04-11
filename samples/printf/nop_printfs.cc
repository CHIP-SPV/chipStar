/*
Copyright (c) 2022 Pekka Jääskeläinen / Parmance for Argonne National Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
 * Test cases for NOP printfs which generate illegal output.
 * The empty string (with a zeroinitializer) is converted to an
 * OpConstantNull which causes (at least with LZ implementation)
 * to print the first printed string again as it happens to be
 * at 0 in the constant AS.
 */

#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "hip/hip_runtime.h"
#include "printf_tests_common.h"

__global__ void nop_str_arg(int *out) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  printf("Howdy HIPpies!\n");
  out[tid] = printf("%s", ""); // Should be NOP except for the retval (1).
  out[tid] += printf("");      // Should be NOP except for the retval (0).
  out[tid] += printf("I'm correct since I'm not an empty string!\n");
}

int main(int argc, char *argv[]) {
  uint num_threads = 1;
  uint failures = 0;

  void *retval_void;
  CHECK(hipHostMalloc(&retval_void, 4 * num_threads), hipSuccess);
  auto retval = reinterpret_cast<int *>(retval_void);

  hipLaunchKernelGGL(nop_str_arg, dim3(1), dim3(1), 0, 0, retval);
  hipStreamSynchronize(0);

  for (uint ii = 0; ii != num_threads; ++ii) {
#ifdef __HIP_PLATFORM_AMD__
    CHECK(retval[ii], strlen(""));
#else
    CHECK(retval[ii], 1);
#endif
  }
  printf((failures == 0) ? "PASSED!\n" : "FAILED!\n");
}
