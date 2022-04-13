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

// A regression test for an infinite recursion in HipGlobalVariablesPass.

#include <hip/hip_runtime.h>
#include "test_common.h"

extern __device__ void *Bar;
__device__ void *Foo = &Bar;
__device__ void *Bar = &Foo;

__global__ void Test(int *Out) { *Out = Foo == &Bar && Bar == &Foo; }

int main() {
  int *IntBufD;
  HIPCHECK(hipMalloc(&IntBufD, sizeof(int)));
  hipLaunchKernelGGL(Test, dim3(1), dim3(1), 0, 0, IntBufD);
  int IntBufH = -1;
  HIPCHECK(hipMemcpy(&IntBufH, IntBufD, sizeof(int), hipMemcpyDeviceToHost));
  HIPASSERT(IntBufH == 1);

  HIPCHECK(hipFree(IntBufD));
  passed();
}
