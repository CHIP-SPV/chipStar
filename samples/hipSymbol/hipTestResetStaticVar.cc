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

// Check static local device variables are reset to their
// initialization value on hipDeviceReset();

#include <hip/hip_runtime.h>
#include "test_common.h"

__global__ void countDown(int *Out) {
  static int Counter = 10;
  *Out = Counter--;
}

int main() {
  int *OutD, OutH;
  HIPCHECK(hipMalloc(&OutD, 1 * sizeof(int)));

  hipLaunchKernelGGL(countDown, dim3(1), dim3(1), 0, 0, OutD);
  HIPCHECK(hipMemcpy(&OutH, OutD, sizeof(int), hipMemcpyDeviceToHost));
  printf("Counter=%d\n", OutH);
  HIPASSERT(OutH == 10);

  hipLaunchKernelGGL(countDown, dim3(1), dim3(1), 0, 0, OutD);
  HIPCHECK(hipMemcpy(&OutH, OutD, sizeof(int), hipMemcpyDeviceToHost));
  HIPASSERT(OutH == 9);

  hipLaunchKernelGGL(countDown, dim3(1), dim3(1), 0, 0, OutD);
  HIPCHECK(hipMemcpy(&OutH, OutD, sizeof(int), hipMemcpyDeviceToHost));
  HIPASSERT(OutH == 8);

  HIPCHECK(hipDeviceReset());

  HIPCHECK(hipMalloc(&OutD, 1 * sizeof(int)));
  hipLaunchKernelGGL(countDown, dim3(1), dim3(1), 0, 0, OutD);
  HIPCHECK(hipMemcpy(&OutH, OutD, sizeof(int), hipMemcpyDeviceToHost));
  HIPASSERT(OutH == 10);

  HIPCHECK(hipFree(OutD));
  passed();
}
