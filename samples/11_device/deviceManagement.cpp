/*
 * Copyright (c) 2021-22 chipStar developers
 * Copyright (c) 2021 Jisheng Zhao <jishengz@gmail.com>
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
#include <cmath>

// hip header file
#include "hip/hip_runtime.h"


#define WIDTH 64

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const int width) {
    __shared__ float sharedMem[WIDTH * WIDTH];

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    sharedMem[y * width + x] = in[x * width + y];

    __syncthreads();

    out[y * width + x] = sharedMem[y * width + x];
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

int main() {
  float* Matrix;
  float* TransposeMatrix;
  float* cpuTransposeMatrix;
  
  float* gpuMatrix;
  float* gpuTransposeMatrix;
  
  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  hipSetDevice(0);
  
  std::cout << "Device name " << devProp.name << std::endl;
  
  int devCount = 0;
  hipGetDeviceCount(&devCount);
  std::cout << "Number of devices: " << devCount << std::endl;
  
  int i = 0;
  int errors = 0;
  
  Matrix = (float*)malloc(NUM * sizeof(float));
  TransposeMatrix = (float*)malloc(NUM * sizeof(float));
  cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));
  
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    Matrix[i] = (float)i * 10.0f;
  }

  // CPU MatrixTranspose computation 
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

  for (int deviceId = 0; deviceId < devCount; deviceId ++) {
    // Set device via ID
    hipSetDevice(deviceId);

    // allocate the memory on the device side    
    hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));
  
    // Memory transfer from host to device
    hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);
    
    // Lauching kernel from host
    hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
		       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
		       gpuMatrix, WIDTH);
  
    // Memory transfer from device to host
    hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);
    
    // verify the results
    float eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
      if (std::fabs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
	printf("%d cpu: %f gpu  %f\n", i, cpuTransposeMatrix[i], TransposeMatrix[i]);
	errors++;
      }
    }

    // free the resources on device side      
    hipFree(gpuMatrix);
    hipFree(gpuTransposeMatrix);
  }

  if (errors != 0) {
    printf("FAILED: %d errors \n", errors);
  } else {
    printf("PASSED!\n");
  }
  
  // free the resources on host side
  free(Matrix);
  free(TransposeMatrix);
  free(cpuTransposeMatrix);
  
  return errors;
}
