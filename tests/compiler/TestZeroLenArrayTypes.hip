// Regression test for "InvalidArraySize: Array size must be at least
// 1: [0 x <type>]" error from llvm-spiv. This error is known to
// appear with this source compiled with -O0.
#include <hip/hip_runtime.h>
// Uses of this are known to generate GEPs with [0 x float] type in
// LLVM IR.
extern __device__ float a[];
extern __device__ float b[];
__global__ void k0(float *x) { *x += a[1]; }
__global__ void k1(float *x, int i) { *x += a[i]; }
__global__ void k2(float *x, bool c) { *x += (c ? a[2] : b[3]); }
__device__ float a[] = {1.0f, 2.0f, 3.0f};
__device__ float b[] = {10.f, 20.f, 30.f};
