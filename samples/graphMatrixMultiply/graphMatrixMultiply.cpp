/*
 * Copyright (c) 2021-22 chipStar developers
 * Copyright (c) 2019 Michal Babej / Tampere University
 * Copyright (c) 2014 Cedric Nugteren, SURFsara
 * https://github.com/CNugteren/myGEMM.git
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

#include <cmath>
#include <cfloat>
#include <cstdlib>

// hip header file
#include "hip/hip_runtime.h"

#include "util.hh"
#include "kernels.hh"

#define SEED 19284975223

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.1920928955078125e-7f
#endif

void verifyResults(float *cpuRes, float *gpuRes) {
  // verify the results
  size_t errors = 0;
  float eps = FLT_EPSILON * 4;
  for (int i = 0; i < WIDTH; i++) {
    for (int j = 0; j < WIDTH; j++) {
      float cpu = cpuRes[i * WIDTH + j];
      float gpu = gpuRes[i * WIDTH + j];
      float diff = std::fabs(gpu - cpu);
      if (diff > std::max(std::fabs(cpu), std::fabs(gpu)) * eps) {
        errors++;
        // if (errors < 10)
        //   std::cout << "E[" << i << "][" << j << "]: M1 "
        //             << Matrix1[i * WIDTH + j] << " M2 "
        //             << Matrix1[i * WIDTH + j] << " CPU: " << cpu
        //             << " GPU: " << gpu << " ERROR: " << diff << "\n";
      }
    }
  }

  if (errors != 0) {
    std::cout << "Verification FAILED: " << errors << "  errors\n";
  } else {
    std::cout << "Verification PASSED!\n";
  }
}

int main(int argc, char **argv) {
  int ITERS = 1;
  if (argc > 1)
    ITERS = atoi(argv[1]);

  hipError_t err;
  std::mt19937 gen(SEED);
  auto rnd =
      std::bind(std::uniform_real_distribution<float>{100.0f, 1000.0f}, gen);

  float *Matrix1;
  float *Matrix2;
  float *MultiplyMatrix;
  float *cpuMultiplyMatrix;

  float *gpuMatrix1;
  float *gpuMatrix2;
  float *gpuMultiplyMatrix;

  hipDeviceProp_t devProp;
  err = hipGetDeviceProperties(&devProp, 0);
  ERR_CHECK;

  std::cout << "Device name " << devProp.name << std::endl;

  hipEvent_t events[ITERS * 2];
  for (uint w = 0; w < (ITERS * 2); w++) {
    err = hipEventCreate(&events[w]);
    ERR_CHECK;
  }

  size_t i, j;

  Matrix1 = new float[NUM];
  Matrix2 = new float[NUM];
  MultiplyMatrix = new float[NUM];
  cpuMultiplyMatrix = new float[NUM];
  for (int i = 0; i < NUM; i++) {
    Matrix1[i] = 0;
    Matrix2[i] = 0;
    MultiplyMatrix[i] = 0;
    cpuMultiplyMatrix[i] = 0;
  }

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    Matrix1[i] = rnd();
    Matrix2[i] = rnd();
  }

  // CPU MatrixMultiply
  auto time1 = std::chrono::high_resolution_clock::now();
  matrixMultiplyCPUReference(Matrix1, Matrix2, cpuMultiplyMatrix);
  auto time2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> fp_ms = time2 - time1;
  std::cout << "matrixMultiplyCPUReference time taken(ms): " << fp_ms.count()
            << "\n";

  // GPU MatrixMultiply
  float minMs, tempMs;

  // allocate the memory on the device side
  err = hipMalloc((void **)&gpuMatrix1, NUM * sizeof(float));
  ERR_CHECK;
  err = hipMalloc((void **)&gpuMatrix2, NUM * sizeof(float));
  ERR_CHECK;
  err = hipMalloc((void **)&gpuMultiplyMatrix, NUM * sizeof(float));
  ERR_CHECK;

  // t1 = stream kernels
  std::cout << "\nRunning " << ITERS
            << " iterations of basic matrix multiply\n";
  auto tMatrixMultiplyBasic =
      matrixMultiplyBasic(gpuMatrix1, gpuMatrix2, Matrix1, Matrix2,
                          gpuMultiplyMatrix, MultiplyMatrix, events, ITERS);
  std::cout << "GPU real time taken(ms): " << tMatrixMultiplyBasic << "\n";
  // verify the results
  verifyResults(cpuMultiplyMatrix, MultiplyMatrix);

  std::cout << "\nRunning " << ITERS
            << " iterations of graph basic matrix multiply\n";
  auto tMatrixMultiplyGraphBasic = matrixMultiplyGraphBasic(
      gpuMatrix1, gpuMatrix2, Matrix1, Matrix2, gpuMultiplyMatrix,
      MultiplyMatrix, events, ITERS);
  std::cout << "GPU real time taken(ms): " << tMatrixMultiplyGraphBasic << "\n";
  // verify the results
  verifyResults(cpuMultiplyMatrix, MultiplyMatrix);

  minMs = 1e30f;
  // for (i = 0; i < ITERS; ++i) {
  //   err = hipEventElapsedTime(&tempMs, events[i * 2], events[i * 2 + 1]);
  //   ERR_CHECK;
  //   std::cout << "hipLaunchKernel " << i << " time taken: " << tempMs <<
  //   "\n"; if (tempMs < minMs)
  //     minMs = tempMs;
  // }

  std::cout << "hipLaunchKernel BEST TIME: " << minMs << "\n";

  for (uint w = 0; w < (ITERS * 2); w++) {
    err = hipEventDestroy(events[w]);
    ERR_CHECK;
  }

  // free the resources on device side
  err = hipFree(gpuMatrix1);
  ERR_CHECK;
  err = hipFree(gpuMatrix2);
  ERR_CHECK;
  err = hipFree(gpuMultiplyMatrix);
  ERR_CHECK;

  // free the resources on host side
  delete[] Matrix1;
  delete[] Matrix2;
  delete[] MultiplyMatrix;
  delete[] cpuMultiplyMatrix;
}
