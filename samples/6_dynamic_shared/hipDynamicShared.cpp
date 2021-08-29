/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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

#include "hip/hip_runtime.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>

#define HIPCHECK(code)                                                         \
  do {                                                                         \
    hiperr = code;                                                             \
    if (hiperr != hipSuccess) {                                                \
      std::cerr << "ERROR on line " << __LINE__ << ": " << (unsigned)hiperr    \
                << "\n";                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

template <typename T>
__global__ void testExternSharedKernel(const T *A_d, const T *B_d, T *C_d,
                                       size_t numElements,
                                       size_t groupElements) {
  // declare dynamic shared memory; should be 2*groupElements size
  HIP_DYNAMIC_SHARED(T, sdata);

  size_t gid = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
  size_t tid = hipThreadIdx_x;
  size_t tid2 = groupElements + hipThreadIdx_x;

  // initialize dynamic shared memory
  if (tid < groupElements) {
    sdata[tid] = static_cast<T>(tid);
    sdata[tid2] = static_cast<T>(tid);
  }

  // prefix sum inside dynamic shared memory

  if (groupElements >= 256) {
    if (tid >= 128 && tid < groupElements) {
      sdata[tid2] = sdata[tid] + sdata[tid - 128];
    }
    __syncthreads();
  }
  if (groupElements >= 128) {
    if (tid >= 64 && tid < groupElements) {
      sdata[tid] = sdata[tid2] + sdata[tid2 - 64];
    }
    __syncthreads();
  }
  if (groupElements >= 64) {
    if (tid >= 32 && tid < groupElements) {
      sdata[tid2] = sdata[tid] + sdata[tid - 32];
    }
    __syncthreads();
  }
  if (groupElements >= 32) {
    if (tid >= 16 && tid < groupElements) {
      sdata[tid] = sdata[tid2] + sdata[tid2 - 16];
    }
    __syncthreads();
  }

  if (groupElements >= 16) {
    if (tid >= 8 && tid < groupElements) {
      sdata[tid2] = sdata[tid] + sdata[tid - 8];
    }
    __syncthreads();
  }

  if (groupElements >= 8) {
    if (tid >= 4 && tid < groupElements) {
      sdata[tid] = sdata[tid2] + sdata[tid2 - 4];
    }
    __syncthreads();
  }

  if (groupElements >= 4) {
    if (tid >= 2 && tid < groupElements) {
      sdata[tid2] = sdata[tid] + sdata[tid - 2];
    }
    __syncthreads();
  }

  if (groupElements >= 2) {
    if (tid >= 1 && tid < groupElements) {
      sdata[tid] = sdata[tid2] + sdata[tid2 - 1];
    }
    __syncthreads();
  }

  __syncthreads();

  C_d[gid] = A_d[gid] + B_d[gid] + sdata[tid % groupElements];
}

template <typename T> size_t testExternShared(size_t N, size_t groupElements) {
  size_t Nbytes = N * sizeof(T);
  hipError_t hiperr = hipSuccess;
  T *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
  T *A_h = nullptr, *B_h = nullptr, *C_h = nullptr;

  A_h = (T *)malloc(N * sizeof(T));
  B_h = (T *)malloc(N * sizeof(T));
  C_h = (T *)malloc(N * sizeof(T));

  HIPCHECK(hipMalloc((void **)&A_d, N * sizeof(T)));
  HIPCHECK(hipMalloc((void **)&B_d, N * sizeof(T)));
  HIPCHECK(hipMalloc((void **)&C_d, N * sizeof(T)));

  for (size_t i = 0; i < N; i++) {
    A_h[i] = (T)i * 2.0f;
    B_h[i] = (T)i * 4.0f;
    C_h[i] = (T)i * 8.0f;
  }

  HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  // calculate the amount of dynamic shared memory required
  size_t groupMemBytes = 2 * groupElements * sizeof(T);
  const int threadsPerBlock = 256;

  // launch kernel with dynamic shared memory
  hipLaunchKernelGGL(testExternSharedKernel<T>,
                     dim3(N / threadsPerBlock), dim3(threadsPerBlock),
                     groupMemBytes, 0, A_d, B_d, C_d, N,
                     groupElements);
  HIPCHECK(hipGetLastError());

  HIPCHECK(hipDeviceSynchronize());

  HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  // verify
  size_t errors = 0;

  for (size_t i = 0; i < N; ++i) {
    size_t tid = (i % groupElements);
    T sumFromSharedMemory = static_cast<T>(tid * (tid + 1) / 2);
    T expected = A_h[i] + B_h[i] + sumFromSharedMemory;

    if (C_h[i] != expected) {
      if (i < 32) {
      std::cerr << std::fixed << std::setprecision(32);
      std::cerr << "At " << i << std::endl;
      std::cerr << "  Computed:" << C_h[i] << std::endl;
      std::cerr << "  Expected:" << expected << std::endl;
      std::cerr << sumFromSharedMemory << std::endl;
      std::cerr << A_h[i] << std::endl;
      std::cerr << B_h[i] << std::endl;
      }
      ++errors;
    }
  }

  free(A_h);
  free(B_h);
  free(C_h);
  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(B_d));
  HIPCHECK(hipFree(C_d));
  return errors;
}

int main(int argc, char *argv[]) {
  size_t errors = 0;

  errors += testExternShared<float>(1024, 4);
  errors += testExternShared<float>(1024, 8);
  errors += testExternShared<float>(1024, 16);
  errors += testExternShared<float>(1024, 32);
  errors += testExternShared<float>(1024, 64);
  errors += testExternShared<float>(1024, 128);
  errors += testExternShared<float>(1024, 256);

  if (errors != 0) {
    std::cout << "hipDynamicShared FAILED: " << errors << " errors\n";
    return 1;
  } else {
    std::cout << "hipDynamicShared PASSED!\n";
    return 0;
  }
}
