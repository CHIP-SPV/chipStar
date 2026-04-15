/*
 * Copyright (c) 2026 chipStar developers
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

// Reproducer for an Intel Graphics Compiler (IGC) issue exposed by libCEED's
// hip-shared-basis-tensor-at-points.h kernels. When several similarly-shaped
// __global__ kernels share the same nested device-function template
// (interp_tensor / interp_transpose with __syncthreads()), IGC's default
// optimization pipeline silently drops some of them from the produced module.
//
// hipModuleGetFunction() then returns hipErrorLaunchFailure for the dropped
// kernel names, even though they were declared as OpEntryPoint in the SPIR-V
// chipStar handed to Level Zero.
//
// This test compiles a synthetic source that triggers the same drop and asserts
// every declared kernel is reachable from the loaded module.

#include "TestCommon.hh"
#include <vector>

static constexpr auto KernelSource = R"---(
// Minimal repro: two helper templates with shared-memory barriers, called from
// six extern "C" __global__ kernels with slightly different bodies. With
// IGC defaults, the kernels named below as "MISSING" disappear from the
// compiled module at -O2; with -cl-opt-disable they are all present.
template<int P, int Q>
__device__ __forceinline__ void contract(double *r_V, const double *r_U,
                                         const double *s_B) {
  for (int i = 0; i < P; ++i) {
    double acc = 0;
    for (int j = 0; j < Q; ++j) acc += r_U[j] * s_B[i * Q + j];
    r_V[i] = acc;
  }
}
template<int P, int Q>
__device__ void interp_tensor(double *r_V, const double *r_U,
                              const double *s_B) {
  double tmp1[Q], tmp2[P];
  contract<Q, P>(tmp1, r_U, s_B);
  __syncthreads();
  contract<P, Q>(tmp2, tmp1, s_B);
  __syncthreads();
  contract<Q, P>(r_V, tmp2, s_B);
}
template<int P, int Q>
__device__ void interp_transpose(double *r_V, const double *r_U,
                                 const double *s_B) {
  double tmp1[Q], tmp2[P];
  contract<P, Q>(tmp1, r_U, s_B);
  __syncthreads();
  contract<Q, P>(tmp2, tmp1, s_B);
  __syncthreads();
  contract<P, Q>(r_V, tmp2, s_B);
}
extern "C" __global__ void Forward(int n, const double *u, double *v,
                                   const double *B) {
  __shared__ double s_B[81];
  for (int i = threadIdx.x; i < 81; i += blockDim.x) s_B[i] = B[i];
  __syncthreads();
  for (int e = blockIdx.x; e < n; e += gridDim.x) {
    double r_U[9], r_V[9];
    for (int i = 0; i < 9; ++i) r_U[i] = u[e * 9 + i];
    interp_tensor<9, 9>(r_V, r_U, s_B);
    for (int i = 0; i < 9; ++i) v[e * 9 + i] = r_V[i];
  }
}
extern "C" __global__ void Transpose(int n, const double *u, double *v,
                                     const double *B) {
  __shared__ double s_B[81];
  for (int i = threadIdx.x; i < 81; i += blockDim.x) s_B[i] = B[i];
  __syncthreads();
  for (int e = blockIdx.x; e < n; e += gridDim.x) {
    double r_U[9], r_V[9];
    for (int i = 0; i < 9; ++i) r_U[i] = u[e * 9 + i];
    interp_transpose<9, 9>(r_V, r_U, s_B);
    for (int i = 0; i < 9; ++i) v[e * 9 + i] = r_V[i];
  }
}
extern "C" __global__ void TransposeAdd(int n, const double *u, double *v,
                                        const double *B) {
  __shared__ double s_B[81];
  for (int i = threadIdx.x; i < 81; i += blockDim.x) s_B[i] = B[i];
  __syncthreads();
  for (int e = blockIdx.x; e < n; e += gridDim.x) {
    double r_U[9], r_V[9];
    for (int i = 0; i < 9; ++i) r_U[i] = u[e * 9 + i];
    interp_transpose<9, 9>(r_V, r_U, s_B);
    for (int i = 0; i < 9; ++i) v[e * 9 + i] += r_V[i];
  }
}
extern "C" __global__ void Forward2(int n, const double *u, double *v,
                                    const double *B) {
  __shared__ double s_B[81];
  for (int i = threadIdx.x; i < 81; i += blockDim.x) s_B[i] = B[i];
  __syncthreads();
  for (int e = blockIdx.x; e < n; e += gridDim.x) {
    double r_U[9], r_V[9];
    for (int i = 0; i < 9; ++i) r_U[i] = u[e * 9 + i];
    interp_tensor<9, 9>(r_V, r_U, s_B);
    for (int i = 0; i < 9; ++i) v[e * 9 + i] = 2 * r_V[i];
  }
}
extern "C" __global__ void Transpose2(int n, const double *u, double *v,
                                      const double *B) {
  __shared__ double s_B[81];
  for (int i = threadIdx.x; i < 81; i += blockDim.x) s_B[i] = B[i];
  __syncthreads();
  for (int e = blockIdx.x; e < n; e += gridDim.x) {
    double r_U[9], r_V[9];
    for (int i = 0; i < 9; ++i) r_U[i] = u[e * 9 + i];
    interp_transpose<9, 9>(r_V, r_U, s_B);
    for (int i = 0; i < 9; ++i) v[e * 9 + i] = 2 * r_V[i];
  }
}
extern "C" __global__ void TransposeAdd2(int n, const double *u, double *v,
                                         const double *B) {
  __shared__ double s_B[81];
  for (int i = threadIdx.x; i < 81; i += blockDim.x) s_B[i] = B[i];
  __syncthreads();
  for (int e = blockIdx.x; e < n; e += gridDim.x) {
    double r_U[9], r_V[9];
    for (int i = 0; i < 9; ++i) r_U[i] = u[e * 9 + i];
    interp_transpose<9, 9>(r_V, r_U, s_B);
    for (int i = 0; i < 9; ++i) v[e * 9 + i] += 2 * r_V[i];
  }
}
)---";

int main() {
  // Compile the source via HIPRTC at default optimization (-O2 internally).
  hiprtcProgram Prog = HiprtcAssertCreateProgram(KernelSource);
  std::vector<const char *> Opts = {"-O2"};
  auto Code = HiprtcAssertCompileProgram(Prog, Opts);

  hipModule_t Module;
  HIP_CHECK(hipModuleLoadData(&Module, Code.data()));

  const char *KernelNames[] = {
      "Forward",  "Transpose",  "TransposeAdd",
      "Forward2", "Transpose2", "TransposeAdd2",
  };

