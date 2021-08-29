// original source:
// Copyright (c) 1993-2016, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cassert>
#include <functional>
#include <iostream>
#include <random>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "fp16_conversion.hpp"

// This is a simple example of using FP16 types and arithmetic on
// GPUs that support it. The code computes an AXPY (A * X + Y) operation
// on half-precision (FP16) vectors (HAXPY).

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
void checkCuda(hipError_t result) {
  if (result != hipSuccess) {
    std::cerr << "CUDA Runtime Error: " << hipGetErrorString(result) << "\n";
    assert(result == hipSuccess);
  }
}

size_t errors;

// TODO investigate: it seems the errors can get quite large (~ 2.0) on some
void check(const half *x, const half *y, const int n) {
  errors = 0;
  float eps = 2.0f;
  for (int i = 0; i < n; i++) {
    float gpu_computed = half_to_float(y[i]);
    float verify = half_to_float(x[i]) * 5.0f;
    if (std::fabs(gpu_computed - verify) > eps) {
      ++errors;
      if (errors < 32) {
        std::cerr << "Error at N: " << i << " x[i]: " << half_to_float(x[i])
                  << " GPU: " << gpu_computed << " CPU: " << verify << "\n";
      }
    }
  }
}

__global__ void haxpy(half a, const half *x, half *y) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  y[i] = x[i] * a;
}

__global__ void haxpy_v2(half a, const half *x, half *y, const int n) {
  int cout = n >> 1;
  int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  if (i < cout ) {
    const half2 *vx = (const half2*)(x) + i;
    const half2 va = {a, a};
    half2* vy = (half2*)(y) + i;
    *vy = *vx * va;
  }
}


#define SEED 923874103841

int main(void) {

  const int n = 33554432;
  const int blockSize = 128;
  const int nBlocks = n / blockSize;

  std::mt19937 gen(SEED);
  auto rnd = std::bind(std::uniform_int_distribution<short>{100, 1000}, gen);

  const half a = approx_float_to_half(5.0f);

  half *x, *y, *d_x, *d_y;
  x = (half *)malloc(n * sizeof(half));
  y = (half *)malloc(n * sizeof(half));

  for (int i = 0; i < n; i++) {
    x[i] = approx_float_to_half(rnd());
    y[i] = approx_float_to_half(16.0f);
  }

  checkCuda(hipMalloc((void **)&d_x, n * sizeof(half)));
  checkCuda(hipMalloc((void **)&d_y, n * sizeof(half)));

  checkCuda(hipMemcpy(d_x, x, n * sizeof(half), hipMemcpyHostToDevice));
  checkCuda(hipMemcpy(d_y, y, n * sizeof(half), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(haxpy, dim3(nBlocks), dim3(blockSize), 0, 0, a, d_x, d_y);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(y, d_y, n * sizeof(half), hipMemcpyDeviceToHost));
  check(x, y, n);

  hipLaunchKernelGGL(haxpy_v2, dim3(nBlocks), dim3(blockSize), 0, 0, a, d_x, d_y, n);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(y, d_y, n * sizeof(half), hipMemcpyDeviceToHost));
  check(x, y, n);

  free(x);
  free(y);
  checkCuda(hipFree(d_x));
  checkCuda(hipFree(d_y));

  if (errors != 0) {
    std::cout << "Verification FAILED: " << errors << "  errors\n";
    return 1;
  } else {
    std::cout << "Verification PASSED!\n";
    return 0;
  }
}
