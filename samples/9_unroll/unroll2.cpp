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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>

// hip header file
#include "hip/hip_runtime.h"


#define WIDTH 64
#define WIDTH2 (WIDTH/4)

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define THREADS_PER_BLOCK_Z 1

__global__ void matrixTransposeUnroll(float* output, const float* input, const int width) {
    int j = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int i = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    #pragma unroll 4
    for (int m = 0; m < 4; m++)
      #pragma unroll 4
      for (int n = 0; n < 4; n++)
        output[(i*4+n) * width + j*4+m] = input[(j*4+m) * width + i*4+n];
}

__global__ void matrixTransposeNoUnroll(float* output, const float* input, const int width) {
    int j = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int i = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    #pragma nounroll
    for (int m = 0; m < 4; m++)
      #pragma nounroll
      for (int n = 0; n < 4; n++)
        output[(i*4+n) * width + j*4+m] = input[(j*4+m) * width + i*4+n];
}


__global__ void matrixTransposeUnroll2(float* output, const float* input, const int width) {
    int j = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int i = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    output[(i*4+0) * width + j*4+0] = input[(j*4+0) * width + i*4+0];
    output[(i*4+1) * width + j*4+1] = input[(j*4+1) * width + i*4+1];
    output[(i*4+2) * width + j*4+2] = input[(j*4+2) * width + i*4+2];
    output[(i*4+3) * width + j*4+3] = input[(j*4+3) * width + i*4+3];
}


__global__ void matrixTranspose(float* output, const float* input, const int width) {
    int j = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int i = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    output[i * width + j] = input[j * width + i];
}


void runTest(float* gpuMatrix, 
             float* gpuTransposeMatrix,
             float* Matrix, 
             float* TransposeMatrix,
	     int  sel) {
    // Memory transfer from host to device
    hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);

    // Lauching kernel from host
    if (sel == 0)
	    hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH/THREADS_PER_BLOCK_X, WIDTH/THREADS_PER_BLOCK_Y), 
			    dim3(THREADS_PER_BLOCK_X , THREADS_PER_BLOCK_Y), 0, 0,
			    gpuTransposeMatrix, gpuMatrix, WIDTH);
    else if (sel == 1)
	    hipLaunchKernelGGL(matrixTransposeNoUnroll, dim3(WIDTH2/THREADS_PER_BLOCK_X, WIDTH2/THREADS_PER_BLOCK_Y), 
			    dim3(THREADS_PER_BLOCK_X , THREADS_PER_BLOCK_Y), 0, 0,
			    gpuTransposeMatrix, gpuMatrix, WIDTH);
    else if (sel == 2)
	    hipLaunchKernelGGL(matrixTransposeUnroll, dim3(WIDTH2/THREADS_PER_BLOCK_X, WIDTH2/THREADS_PER_BLOCK_Y), 
			    dim3(THREADS_PER_BLOCK_X , THREADS_PER_BLOCK_Y), 0, 0,
			    gpuTransposeMatrix, gpuMatrix, WIDTH);
    else
	    hipLaunchKernelGGL(matrixTransposeUnroll2, dim3(WIDTH2/THREADS_PER_BLOCK_X, WIDTH2/THREADS_PER_BLOCK_Y), 
			    dim3(THREADS_PER_BLOCK_X , THREADS_PER_BLOCK_Y), 0, 0,
			    gpuTransposeMatrix, gpuMatrix, WIDTH);

    // Memory transfer from device to host
    hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);
}

int verify(const float* TransposeMatrix, const float* cpuTransposeMatrix) {
    int errors = 0;
    for (int i = 0; i < NUM; i++) {
        if (TransposeMatrix[i] != cpuTransposeMatrix[i]) {
            printf("%d cpu: %f gpu  %f\n", i, cpuTransposeMatrix[i], TransposeMatrix[i]);
            errors++;
	    break;
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }
    return errors;
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

    std::cout << "Device name " << devProp.name << std::endl;

    int i;
    int errors;

    Matrix = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix = (float*)malloc(NUM * sizeof(float));
    cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix[i] = (float)i;
    }

    matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

    runTest(gpuMatrix, gpuTransposeMatrix, Matrix, TransposeMatrix, 0);
    errors = verify(TransposeMatrix, cpuTransposeMatrix);

    runTest(gpuMatrix, gpuTransposeMatrix, Matrix, TransposeMatrix, 1);
    errors += verify(TransposeMatrix, cpuTransposeMatrix);

    runTest(gpuMatrix, gpuTransposeMatrix, Matrix, TransposeMatrix, 2);
    errors += verify(TransposeMatrix, cpuTransposeMatrix);

    runTest(gpuMatrix, gpuTransposeMatrix, Matrix, TransposeMatrix, 3);
    errors += verify(TransposeMatrix, cpuTransposeMatrix);

    // free the resources on device side
    hipFree(gpuMatrix);
    hipFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

    return errors;
}
