/*
Copyright (c) 2022 Henry Linjamäki / Parmance for Argonne National Laboratory

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
// Check abort functionality on a kernel called with <<<>>>-syntax.
#include "hip/hip_runtime.h"

#include <stdio.h>

__global__ void abort_kernel(int *Out) {
  abort();
  *Out = 123;
}

int main(int argc, char *argv[]) {
  unsetenv("CHIP_HOST_IGNORES_DEVICE_ABORT"); // Force abort behavior.
  int *OutD;
  hipError_t Err = hipMalloc(&OutD, sizeof(int));
  assert(Err == hipSuccess);
  abort_kernel<<<dim3(1), dim3(1)>>>(OutD);
  // Control should not reach here.
  printf("Error: abort() was ignored!\n");
  return 0;
}
