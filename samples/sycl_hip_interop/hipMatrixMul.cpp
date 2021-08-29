#include <iostream>
#include <random>
#include <functional>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cfloat>
   
// hip header file
#include "hip/hip_runtime.h"

#include "sycl_hiplz_interop.h"
 
#define MM_SHARED

#define WIDTH 16 // 1024
// the required shared memory is (2 * 4 * THREADS_PER_BLOCK * THREADS_PER_BLOCK) bytes
#define THREADS_PER_BLOCK 16

#define ERR_CHECK_2 \
  do { \
  err = hipGetLastError(); \
    if (err != hipSuccess) { \
      std::cerr << "HIP API error\n"; \
      return -1; \
    } \
  } while (0)


#define ERR_CHECK \
  do { \
    if (err != hipSuccess) { \
      std::cerr << "HIP API error\n"; \
      return -1; \
    } \
  } while (0)


__global__ void gpuMatrixMul(const float * __restrict A,
			     const float * __restrict B,
			     float * __restrict C,
			     uint M, uint N, uint K) {
  // Thread identifiers
  const uint globalRow = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; // Row ID of C (0..M)
  const uint globalCol = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y; // Col ID of C (0..N)

  // Compute a single element (loop over K)
  float acc = 0.0f;
  for (uint k = 0; k < K; k++) {
    acc += A[k * M + globalRow] * B[globalCol * K + k];
  }

  // Store the result
  C[globalCol * M + globalRow] = acc;
}

/*****************************************************************************/

int hiplzInit(void* driverPtr, void* deviePtr, void* contextPtr, void* queuePtr) {
  hipError_t err = hipInitFromOutside(driverPtr, deviePtr, contextPtr, queuePtr);
  ERR_CHECK;
  
  return 0;
}

int hipMatrixMultiplicationUSMTest(const float* A, const float* B, float* C, int M, int N) {

  hipError_t err;

  hipDeviceProp_t devProp;
  err = hipGetDeviceProperties(&devProp, 0);
  ERR_CHECK;

  // Lauching kernel from host
  hipLaunchKernelGGL(gpuMatrixMul,
		     dim3(WIDTH / THREADS_PER_BLOCK, WIDTH / THREADS_PER_BLOCK),
		     dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK),
		     0, 0,
		     A, B, C, M, N, M);
  ERR_CHECK_2;
  
  err = hipDeviceSynchronize();
  ERR_CHECK;
  
  return 0;
}

int hipMatrixMultiplicationTest(const float* A, const float* B, float* C, int M, int N) {

  hipError_t err;

  hipDeviceProp_t devProp;
  err = hipGetDeviceProperties(&devProp, 0);
  ERR_CHECK;

  float* gpuMatrix1;
  float* gpuMatrix2;
  float* gpuMultiplyMatrix;
  
  int NUM = M * N;
  // allocate the memory on the device side  
  err = hipMalloc((void**)&gpuMatrix1, NUM * sizeof(float));
  ERR_CHECK;
  err = hipMalloc((void**)&gpuMatrix2, NUM * sizeof(float));
  ERR_CHECK;
  err = hipMalloc((void**)&gpuMultiplyMatrix, NUM * sizeof(float));
  ERR_CHECK;

  // Memory transfer from host to device   
  err = hipMemcpy(gpuMatrix1, A, NUM * sizeof(float), hipMemcpyHostToDevice);
  ERR_CHECK;

  err = hipMemcpy(gpuMatrix2, B, NUM * sizeof(float), hipMemcpyHostToDevice);
  ERR_CHECK;
  
  // Lauching kernel from host 
  hipLaunchKernelGGL(gpuMatrixMul,
                     dim3(WIDTH / THREADS_PER_BLOCK, WIDTH / THREADS_PER_BLOCK),
                     dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK),
                     0, 0,
                     gpuMatrix1, gpuMatrix2, gpuMultiplyMatrix, M, N, M);
  ERR_CHECK_2;

  // Memory transfer from device to host   
  err = hipMemcpy(C, gpuMultiplyMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);
  ERR_CHECK;

  err = hipDeviceSynchronize();
  ERR_CHECK;

  return 0;
}

