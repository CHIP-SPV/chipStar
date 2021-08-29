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


#define SG_SIZE 8
#define NUM_G 8
#define TOTAL (NUM_G * SG_SIZE)

#define HIPCHECK(code)                                                         \
  do {                                                                         \
    hiperr = code;                                                             \
    if (hiperr != hipSuccess) {                                                \
      std::cerr << "ERROR on line " << __LINE__ << ": " << (unsigned)hiperr    \
                << "\n";                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in) {
    unsigned b_start = hipBlockDim_x * hipBlockIdx_x;
    unsigned b_offs = b_start + hipThreadIdx_x;
    unsigned s_offs = hipBlockDim_x - hipThreadIdx_x - 1;

    float val = in[b_offs];

    out[b_offs] = __shfl(val, s_offs);
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input) {
    for (unsigned i = 0; i < NUM_G; ++i) {
        for (unsigned j = 0; j < SG_SIZE; j++) {
            output[i * SG_SIZE + j] = input[i * SG_SIZE + SG_SIZE - j - 1];
        }
    }
}

int main() {
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;

    hipError_t hiperr = hipSuccess;

    hipDeviceProp_t devProp;
    HIPCHECK(hipGetDeviceProperties(&devProp, 0));

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
    HIPCHECK(hipMalloc((void **)&gpuMatrix, TOTAL * sizeof(float)));
    HIPCHECK(hipMalloc((void **)&gpuTransposeMatrix, TOTAL * sizeof(float)));

    // Memory transfer from host to device
    HIPCHECK(hipMemcpy(gpuMatrix, Matrix, TOTAL * sizeof(float),
                       hipMemcpyHostToDevice));

    // Lauching kernel from host
    hipLaunchKernelGGL(matrixTranspose, dim3(NUM_G), dim3(SG_SIZE), 0, 0,
                       gpuTransposeMatrix, gpuMatrix);
    HIPCHECK(hipGetLastError());

    // Memory transfer from device to host
    HIPCHECK(hipMemcpy(TransposeMatrix, gpuTransposeMatrix,
                       TOTAL * sizeof(float), hipMemcpyDeviceToHost));

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
      HIPCHECK(hipFree(gpuMatrix));
      HIPCHECK(hipFree(gpuTransposeMatrix));

      // free the resources on host side
      free(Matrix);
      free(TransposeMatrix);
      free(cpuTransposeMatrix);

      return errors;
}
