/*
 * Copyright (c) 2021-22 chipStar developers
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


#include "hip/hip_runtime.h"
#include <cstdio>

__global__ void kernel(int *Out, int *In) { *Out = *In; }

int main() {
  int InH = 123, OutH = 0, *InD, *OutD;
  (void)hipMalloc(&OutD, sizeof(int));
  (void)hipMalloc(&InD, sizeof(int));
  (void)hipMemcpy(InD, &InH, sizeof(int), hipMemcpyHostToDevice);
  kernel<<<1, 1>>>(OutD, InD);
  (void)hipMemcpy(&OutH, OutD, sizeof(int), hipMemcpyDeviceToHost);
  printf("OutH=%d\n", OutH);
  if (OutH == 123) {
    printf("PASSED\n");
    return 0;
  }
  return 1;
}
