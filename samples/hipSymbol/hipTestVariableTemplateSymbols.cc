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

#include <hip/hip_runtime.h>
#include "test_common.h"

template <typename T> __device__ T Foo(12);
template <> __device__ char Foo<char> = 34;

//__device__ int *FooPtr;

__global__ void ReadFoo(int *Out) {
  Foo<int> += 1;
  Out[0] = Foo<long long>;
  Out[1] = Foo<int>; // Note: The Foo<int> here and two lines above
                     // are the same instance.
  Out[2] = Foo<char>;
  Foo<char> = 0;
}

int main() {
  int *OutD, OutH[3];
  HIPCHECK(hipMalloc(&OutD, 3 * sizeof(int)));

  hipLaunchKernelGGL(ReadFoo, dim3(1), dim3(1), 0, 0, OutD);
  HIPCHECK(hipMemcpy(&OutH, OutD, 3 * sizeof(int), hipMemcpyDeviceToHost));
  // printf("OutH = {0x%X, 0x%X, 0x%X}\n", OutH[0], OutH[1], OutH[2]);
  HIPASSERT(OutH[0] == 12);
  HIPASSERT(OutH[1] == 13);
  HIPASSERT(OutH[2] == 34);

  hipLaunchKernelGGL(ReadFoo, dim3(1), dim3(1), 0, 0, OutD);
  HIPCHECK(hipMemcpy(&OutH, OutD, 3 * sizeof(int), hipMemcpyDeviceToHost));
  // printf("OutH = {0x%X, 0x%X, 0x%X}\n", OutH[0], OutH[1], OutH[2]);
  HIPASSERT(OutH[0] == 12);
  HIPASSERT(OutH[1] == 14);
  HIPASSERT(OutH[2] == 0);

  HIPCHECK(hipDeviceReset());

  HIPCHECK(hipMalloc(&OutD, 3 * sizeof(int)));
  hipLaunchKernelGGL(ReadFoo, dim3(1), dim3(1), 0, 0, OutD);
  HIPCHECK(hipMemcpy(&OutH, OutD, 3 * sizeof(int), hipMemcpyDeviceToHost));
  // printf("OutH = {0x%X, 0x%X, 0x%X}\n", OutH[0], OutH[1], OutH[2]);
  HIPASSERT(OutH[0] == 12);
  HIPASSERT(OutH[1] == 13);
  HIPASSERT(OutH[2] == 34);

  HIPCHECK(hipFree(OutD));
  passed();
}
