/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANUMTY OF ANY KIND, EXPRESS OR
IMPLIED, INUMCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNUMESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANUMY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INUM AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INUM CONUMECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <mutex>
#include <cmath>

#include "hip/hip_runtime.h"

#define WIDTH 32

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

using namespace std;

void TestCallback(hipStream_t stream, hipError_t status, void* userData) {
  std::cout << "Invoke CALLBACK\n";
}

int main() {

  float *data[2], *TransposeMatrix[2], *gpuTransposeMatrix[2], *randArray;
  int width = WIDTH;
  randArray = (float*)malloc(NUM * sizeof(float));
  TransposeMatrix[0] = (float*)calloc(NUM , sizeof(float));
  hipMalloc((void**)&gpuTransposeMatrix[0], NUM * sizeof(float));

    hipMalloc((void**)&data[0], NUM * sizeof(float));
    hipMemcpyAsync(TransposeMatrix[0], gpuTransposeMatrix[0], NUM * sizeof(float),
                      hipMemcpyDeviceToHost, 0);
    hipStreamAddCallback(0, TestCallback, (void* )TransposeMatrix[0], 0);
    printf("hipDeviceSync()\n");
    hipDeviceSynchronize();

    printf("stream PASSED!\n");
    return 0;
}
