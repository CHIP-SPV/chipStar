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
#include <cmath>

#include "hip/hip_runtime.h"

#define WIDTH 4

#define TOTAL (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    float val = in[y * WIDTH + x];

    out[x * WIDTH + y] = __shfl(val, y * WIDTH + x);
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input) {
    for (unsigned int j = 0; j < WIDTH; j++) {
        for (unsigned int i = 0; i < WIDTH; i++) {
            output[i * WIDTH + j] = input[j * WIDTH + i];
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

    Matrix = (float*)malloc(TOTAL * sizeof(float));
    TransposeMatrix = (float*)malloc(TOTAL * sizeof(float));
    cpuTransposeMatrix = (float*)malloc(TOTAL * sizeof(float));

    // initialize the input data
    for (i = 0; i < TOTAL; i++) {
        Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatrix, TOTAL * sizeof(float));
    hipMalloc((void**)&gpuTransposeMatrix, TOTAL * sizeof(float));

    // Memory transfer from host to device
    hipMemcpy(gpuMatrix, Matrix, TOTAL * sizeof(float), hipMemcpyHostToDevice);

    // Lauching kernel from host
    hipLaunchKernelGGL(matrixTranspose, dim3(1, 1), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0,
                    gpuTransposeMatrix, gpuMatrix);

    // Memory transfer from device to host
    hipMemcpy(TransposeMatrix, gpuTransposeMatrix, TOTAL * sizeof(float), hipMemcpyDeviceToHost);

    // CPU MatrixTranspose computation
    matrixTransposeCPUReference(cpuTransposeMatrix, Matrix);

    // verify the results
    errors = 0;
    float eps = 1.0E-6;
    for (i = 0; i < TOTAL; i++) {
        if (std::fabs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
            std::cout << "ITEM: " << i <<
                         " cpu: " << cpuTransposeMatrix[i] <<
                         " gpu: " << TransposeMatrix[i] << "\n";
            errors++;
        }
        else
          std::cout << "ITEM " << i << " OK\n";
    }

    if (errors > 0) {
        std::cout << "FAIL: " << errors << " errors \n";
      }
    else {
        std::cout << "PASSED\n";
      }



    // free the resources on device side
    hipFree(gpuMatrix);
    hipFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

    return errors;
}
