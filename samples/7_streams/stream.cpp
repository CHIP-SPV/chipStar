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

// #define NUM  256 // pass
 #define NUM  257 // hangs

using namespace std;

int id1, id2;
void TestCallback(hipStream_t stream, hipError_t status, void* userData) {
  std::cout << "Invoke CALLBACK " << std::endl;
}


  //hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float),
  //               hipMemcpyDeviceToHost); // pass
  //hipMemcpyAsync(gpuTransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), 
  //               hipMemcpyDeviceToDevice, 0); // pass
  //hipMemcpyAsync(gpuTransposeMatrix, TransposeMatrix, NUM * sizeof(float), 
  //               hipMemcpyHostToDevice, 0); // pass

int main() {
  float *TransposeMatrix, *gpuTransposeMatrix, *randArray;
  TransposeMatrix = (float*)calloc(NUM , sizeof(float));
  hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

  hipMemcpyAsync(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float),
                 hipMemcpyDeviceToHost, 0); // fail
  hipStreamAddCallback(0, TestCallback, nullptr, 0);
  std::cout << "Callback enqueue done\n";
  hipDeviceSynchronize();
  std::cout << "hipDeviceSync done\n";

  return 0;
}
