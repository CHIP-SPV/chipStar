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
#include <string>
#include <iomanip>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "fp16_conversion.hpp"

// This is a simple example of using FP16 types and arithmetic on
// GPUs that support it. The code computes an AXPY (A * X + Y) operation
// on half-precision (FP16) vectors (HAXPY).

// Convenience function for checking HIP runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
void checkCuda(hipError_t result) {
  if (result != hipSuccess) {
    std::cerr << "HIP Runtime Error: " << hipGetErrorString(result) << "\n";
    //assert(result == hipSuccess);
  }
}

size_t errors;

// TODO investigate: it seems the errors can get quite large (~ 2.0) on some
void check(std::string fn, const half *x, const half *y, const half *z, const int n) {
  size_t current_errors = 0;

  bool eq_oper = false;
  if ((fn == "eq") || (fn == "neq") || (fn == "lt") ||
     (fn == "gt") || (fn == "le") || (fn == "ge"))
    eq_oper = true;

  for (int i = 0; i < n; i++) {
    float xx_computed = half_to_float(x[i]);
    float yy_computed = half_to_float(y[i]);
    float zz_computed = half_to_float(z[i]);

    float verify;
    if (fn == "sub") 
	    verify = xx_computed - yy_computed;
    else if (fn == "add")
	    verify = xx_computed + yy_computed;
    else if (fn == "mul")
	    verify = xx_computed * yy_computed;
    else if (fn == "div")
	    verify = xx_computed / yy_computed;
    else if (fn == "fma")
	    verify = xx_computed * yy_computed + yy_computed;
    else if (fn == "neg")
	    verify = -1.0f * xx_computed;
    else if (fn == "eq")
	    verify = xx_computed == yy_computed;
    else if (fn == "neq")
	    verify = xx_computed != yy_computed;
    else if (fn == "lt")
	    verify = xx_computed < yy_computed;
    else if (fn == "gt")
	    verify = xx_computed > yy_computed;
    else if (fn == "le")
	    verify = xx_computed <= yy_computed;
    else if (fn == "ge")
	    verify = xx_computed >= yy_computed;

    if ((eq_oper && (zz_computed != verify)) ||
        (!eq_oper && compare_calculated(zz_computed, verify) > 4)) {
      if (current_errors < 8) {
        std::cerr << std::hexfloat <<"Test " << fn << " failed at : " << i << " x[i]: "
		  << xx_computed << " y[i]: " << xx_computed
		  << " GPU: " << zz_computed
		  << " CPU: " << verify << "\n";
      }
      ++current_errors;
    }
  }

  errors += current_errors;
  std::cerr << "Test " << fn << " errors: " << current_errors << "\n";
}

__global__ void half_add(const half *x, const half *y, half *z) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  z[i] = __hadd(x[i], y[i]);
}

__global__ void half_neg(const half *x, const half *y, half *z) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  z[i] = __hneg(x[i]);
}

__global__ void half_sub(const half *x, const half *y, half *z) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  z[i] = __hsub(x[i], y[i]);
}

__global__ void half_fma(const half *x, const half *y, half* z) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  z[i] = __hfma(x[i], y[i], y[i]);
}

__global__ void half_mul(const half *x, const half *y, half *z) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  z[i] = __hmul(x[i], y[i]);
}

__global__ void half_div(const half *x, const half *y, half *z) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  z[i] = __hdiv(x[i], y[i]);
}

__global__ void half_eq(const half *x, const half *y, half *z) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  z[i] = __heq(x[i], y[i]);
}

__global__ void half_ge(const half *x, const half *y, half *z) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  z[i] = __hge(x[i], y[i]);
}

__global__ void half_gt(const half *x, const half *y, half *z) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  z[i] = __hgt(x[i], y[i]);
}

__global__ void half_le(const half *x, const half *y, half *z) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  z[i] = __hle(x[i], y[i]);
}

__global__ void half_lt(const half *x, const half *y, half *z) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  z[i] = __hlt(x[i], y[i]);
}
__global__ void half_ne(const half *x, const half *y, half *z) {
  size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  z[i] = __hne(x[i], y[i]);
}



int main(void) {

  const int n = 65536;
  const int blockSize = 128;
  const int nBlocks = n / blockSize;

  std::random_device source;
  std::mt19937 gen(source());
  auto rnd = std::bind(std::uniform_int_distribution<short>{-100, 100}, gen);

  half *x, *y, *z, *d_x, *d_y, *d_z;
  x = (half *)malloc(n * sizeof(half));
  y = (half *)malloc(n * sizeof(half));
  z = (half *)malloc(n * sizeof(half));

  for (int i = 0; i < n; i++) {
    x[i] = __half(static_cast<__half>(rnd()));
    y[i] = __half(static_cast<__half>(rnd()));
  }

  checkCuda(hipMalloc((void **)&d_x, n * sizeof(half)));
  checkCuda(hipMalloc((void **)&d_y, n * sizeof(half)));
  checkCuda(hipMalloc((void **)&d_z, n * sizeof(half)));

  checkCuda(hipMemcpy(d_x, x, n * sizeof(half), hipMemcpyHostToDevice));
  checkCuda(hipMemcpy(d_y, y, n * sizeof(half), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(half_sub, dim3(nBlocks), dim3(blockSize), 0, 0, d_x, d_y, d_z);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(z, d_z, n * sizeof(half), hipMemcpyDeviceToHost));
  check("sub", x, y, z, n);

  hipLaunchKernelGGL(half_add, dim3(nBlocks), dim3(blockSize), 0, 0, d_x, d_y, d_z);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(z, d_z, n * sizeof(half), hipMemcpyDeviceToHost));
  check("add", x, y, z, n);

  hipLaunchKernelGGL(half_mul, dim3(nBlocks), dim3(blockSize), 0, 0, d_x, d_y, d_z);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(z, d_z, n * sizeof(half), hipMemcpyDeviceToHost));
  check("mul", x, y, z, n);

  hipLaunchKernelGGL(half_div, dim3(nBlocks), dim3(blockSize), 0, 0, d_x, d_y, d_z);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(z, d_z, n * sizeof(half), hipMemcpyDeviceToHost));
  check("div", x, y, z, n);

  hipLaunchKernelGGL(half_fma, dim3(nBlocks), dim3(blockSize), 0, 0, d_x, d_y, d_z);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(z, d_z, n * sizeof(half), hipMemcpyDeviceToHost));
  check("fma", x, y, z, n);

  hipLaunchKernelGGL(half_neg, dim3(nBlocks), dim3(blockSize), 0, 0, d_x, d_y, d_z);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(z, d_z, n * sizeof(half), hipMemcpyDeviceToHost));
  check("neg", x, y, z, n);

  hipLaunchKernelGGL(half_eq, dim3(nBlocks), dim3(blockSize), 0, 0, d_x, d_y, d_z);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(z, d_z, n * sizeof(half), hipMemcpyDeviceToHost));
  check("eq", x, y, z, n);

  hipLaunchKernelGGL(half_ge, dim3(nBlocks), dim3(blockSize), 0, 0, d_x, d_y, d_z);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(z, d_z, n * sizeof(half), hipMemcpyDeviceToHost));
  check("ge", x, y, z, n);

  hipLaunchKernelGGL(half_gt, dim3(nBlocks), dim3(blockSize), 0, 0, d_x, d_y, d_z);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(z, d_z, n * sizeof(half), hipMemcpyDeviceToHost));
  check("gt", x, y, z, n);

  hipLaunchKernelGGL(half_le, dim3(nBlocks), dim3(blockSize), 0, 0, d_x, d_y, d_z);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(z, d_z, n * sizeof(half), hipMemcpyDeviceToHost));
  check("le", x, y, z, n);

  hipLaunchKernelGGL(half_lt, dim3(nBlocks), dim3(blockSize), 0, 0, d_x, d_y, d_z);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(z, d_z, n * sizeof(half), hipMemcpyDeviceToHost));
  check("lt", x, y, z, n);

  hipLaunchKernelGGL(half_ne, dim3(nBlocks), dim3(blockSize), 0, 0, d_x, d_y, d_z);
  checkCuda(hipGetLastError());
  checkCuda(hipMemcpy(z, d_z, n * sizeof(half), hipMemcpyDeviceToHost));
  check("neq", x, y, z, n);

  free(x);
  free(y);
  free(z);
  checkCuda(hipFree(d_x));
  checkCuda(hipFree(d_y));
  checkCuda(hipFree(d_z));

  if (errors != 0) {
    std::cout << "Verification FAILED: " << errors << "  errors\n";
    return 1;
  } else {
    std::cout << "Verification PASSED!\n";
    return 0;
  }
}
