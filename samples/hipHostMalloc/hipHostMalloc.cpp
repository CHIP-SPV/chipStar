/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 * Copyright (c) 2019 Michal Babej / Tampere University
 * Copyright (c) 2014 Cedric Nugteren, SURFsara https://github.com/CNugteren/myGEMM.git
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

#include <iostream>
#include <random>
#include <functional>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cfloat>

// hip header file
#include "hip/hip_runtime.h"

__global__ void setOne(int *A) {
    A[0] = 1;
}

int main() {
  // test hipHostMalloc
  int *hostPtr = nullptr;
  hipError_t Status;
  Status = hipHostMalloc(&hostPtr, 1024 * sizeof(int), hipHostMallocDefault);
  assert(Status == hipSuccess);
  hostPtr[0] = 0;

  hipLaunchKernelGGL(setOne, dim3(1), dim3(1), 0, 0, hostPtr);
  Status = hipGetLastError();
  assert(Status == hipSuccess);
  hipDeviceSynchronize();

  if(hostPtr[0] == 1) {
    std::cout << "PASSED" << std::endl;
  } else {
    std::cout << "FAILED" << std::endl;
  }


}
