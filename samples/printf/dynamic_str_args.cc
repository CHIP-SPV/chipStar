/*
 * Copyright (c) 2021-22 chipStar developers
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
 * Test cases for %s with dynamic arguments.
 */

#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "hip/hip_runtime.h"
#include "printf_tests_common.h"

__global__ void conditional_string(int *io) {
  printf("## conditional string\n");

  const char *dyn_a = "inout[0] was 1";
  const char *dyn_b = "inout[0] was something else than 1, it was";
  const char *str = nullptr;

  if (io[0] == 1) {
    str = dyn_a;
  } else {
    str = dyn_b;
  }
  io[0] = printf("%s %d\n", str, io[0]);
}

__global__ void array_of_strings(int *io) {
  printf("## array of strings, selecting index %d\n", io[0]);

  const char *dyn_a = "I am a dynamic str arg A.\n";
  const char *dyn_b = "I am a dynamic str arg B.\n";

  const char *array_of_strings[] = {dyn_a, dyn_b};

  io[0] = printf("%s\n", array_of_strings[io[0]]);
}

__global__ void host_defined_strings(int *io, const char *str) {
  io[0] = printf("## host defined string: %s", str);
  io[0] += printf("## host defined string with skip: %s", str + 5);
}

int main(int argc, char *argv[]) {
  uint num_threads = 1;
  uint failures = 0;

  void *retval_void;
  CHECK(hipHostMalloc(&retval_void, 4 * num_threads), hipSuccess);
  auto io = reinterpret_cast<int *>(retval_void);

  io[0] = 1;
  hipLaunchKernelGGL(conditional_string, dim3(1), dim3(1), 0, 0, io);
  hipStreamSynchronize(0);

  // Just smoke check the return values, the std output is verified by ctest.
  CHECK_GT(io[0], 1);

  io[0] = 1;

  hipLaunchKernelGGL(array_of_strings, dim3(1), dim3(1), 0, 0, io);
  hipStreamSynchronize(0);

  CHECK_GT(io[0], 0);

  const char *TrulyDyn = "Extremely dynamic printf str!\n";
  char *TrulyDynB;
  hipHostMalloc(&TrulyDynB, strlen(TrulyDyn) + 1);
  memcpy(TrulyDynB, TrulyDyn, strlen(TrulyDyn) + 1);

  hipLaunchKernelGGL(host_defined_strings, dim3(1), dim3(1), 0, 0, io,
                     TrulyDynB);
  hipStreamSynchronize(0);

  CHECK_GT(io[0], 1);

  printf((failures == 0) ? "PASSED!\n" : "FAILED!\n");
  return (failures == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
