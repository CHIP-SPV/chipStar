/*
 * Copyright (c) 2021-22 chipStar developers
 * Copyright (c) 2021-22 Paulius Velesko <paulius.velesko@intel.com>
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

#include "hip/hip_runtime.h"

// STL classes
#include <exception>
#include <iostream>

#include <vector>

#include "hip/hip_interop.h"

#include "hip_sycl_interop.h"

using namespace std;

const int WIDTH = 10;

bool ValueSame(float a, float b) { return std::fabs(a - b) < 1.0e-08; }

bool VerifyResult(float *c_A, float *c_B) {
  bool MismatchFound = false;

  for (size_t i = 0; i < WIDTH; i++) {
    for (size_t j = 0; j < WIDTH; j++) {
      if (!ValueSame(c_A[i * WIDTH + j], c_B[i * WIDTH + j])) {
        std::cout << "fail - The result is incorrect for element: [" << i
                  << ", " << j << "], expected: " << c_A[i * WIDTH + j]
                  << " , but got: " << c_B[i * WIDTH + j] << std::endl;

        MismatchFound = true;
      }
    }
  }

  if (!MismatchFound) {
    std::cout << "SUCCESS - The results are correct!" << std::endl;
  }
  return MismatchFound;  
}

int main() {
  float *A = (float *)malloc(WIDTH * WIDTH * sizeof(float));
  float *B = (float *)malloc(WIDTH * WIDTH * sizeof(float));
  float *C = (float *)malloc(WIDTH * WIDTH * sizeof(float));
  float *C_serial = (float *)malloc(WIDTH * WIDTH * sizeof(float));
  int m, n, k;
  m = n = k = WIDTH;
  int ldA, ldB, ldC;
  ldA = ldB = ldC = WIDTH;
  float alpha = 1.0;
  float beta = 0.0;

  // initialize data on the host
  // prepare matrix data with ROW-major style
  // A(M, N)
  for (size_t i = 0; i < WIDTH; i++)
    for (size_t j = 0; j < WIDTH; j++)
      A[i * WIDTH + j] = i * WIDTH + j;
  // B(N, P)
  for (size_t i = 0; i < WIDTH; i++)
    for (size_t j = 0; j < WIDTH; j++)
      B[i * WIDTH + j] = i * WIDTH + j;

  // get CPU result for verification
  // Resultant matrix: C_serial = A*B
  for (size_t i = 0; i < WIDTH; i++) {
    for (size_t j = 0; j < WIDTH; j++) {
      C_serial[i * WIDTH + j] = 0;
      for (size_t d = 0; d < WIDTH; d++) {
        C_serial[i * WIDTH + j] += A[i * WIDTH + d] * B[d * WIDTH + j];
      }
    }
  }

#define CHECK(err)                                                             \
  do {                                                                         \
    if (err != hipSuccess) {                                                   \
      std::cout << hipGetErrorString(err) << std::endl;                        \
      std::abort();                                                            \
    }                                                                          \
  } while (0);

  hipStream_t stream = nullptr;
  hipError_t error;
  error = hipStreamCreate(&stream);
  CHECK(error);

  assert(stream != nullptr);

  int numHandles;
  error = (hipError_t)hipGetBackendNativeHandles(0, 0, &numHandles);
  CHECK(error);
  uintptr_t nativeHandlers[numHandles];
  error = (hipError_t)hipGetBackendNativeHandles((uintptr_t)stream,
                                                 nativeHandlers, 0);
  CHECK(error);

  // Invoke oneMKL GEEM
  oneMKLGemmTest(nativeHandlers, A, B, C, m, m, k, ldA, ldB, ldC,
                 alpha, beta);

  // check results
  std::cout << "Verify results between OneMKL & Serial: ";
  bool MismatchFound=VerifyResult(C, C_serial);
  if (MismatchFound)
    return 1;
  else
    return 0;
}
