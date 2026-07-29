// Reproduces: __hip_atomic_fetch_min / __hip_atomic_fetch_max on a floating
// point type lower to `atomicrmw fmin` / `fmax`, which llvm-spirv can only
// translate with SPV_EXT_shader_atomic_float_min_max. That extension is not in
// chipStar's allow list, so the program does not even link:
//
//   RequiresExtension: Feature requires the following SPIR-V extension:
//    SPV_EXT_shader_atomic_float_min_max
//   clang++: error: hipspv-link command failed with exit code 18
//
// This is a different path from atomicMin(float *, float), which chipStar
// routes to the CAS-based __chip_atomic_min_f32 in devicelib and which already
// works (see TestAtomicMinMaxFloat.cpp).

#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(X)                                                               \
  do {                                                                         \
    hipError_t E = (X);                                                        \
    if (E != hipSuccess) {                                                     \
      printf("FAILED: %s -> %s\n", #X, hipGetErrorString(E));                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

constexpr int NumThreads = 256;

__global__ void reduceFloat(float *Min, float *Max) {
  float V = static_cast<float>(threadIdx.x) - 100.0f;
  __hip_atomic_fetch_min(Min, V, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_fetch_max(Max, V, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__global__ void reduceDouble(double *Min, double *Max) {
  double V = static_cast<double>(threadIdx.x) - 100.0;
  __hip_atomic_fetch_min(Min, V, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_fetch_max(Max, V, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// The return value is the value seen before the update, so a single thread
// walking a known sequence pins down the semantics exactly.
__global__ void returnsPreviousValue(float *Cell, float *Seen) {
  Seen[0] = __hip_atomic_fetch_min(Cell, 5.0f, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT); // was 10, now 5
  Seen[1] = __hip_atomic_fetch_min(Cell, 7.0f, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT); // stays 5
  Seen[2] = __hip_atomic_fetch_max(Cell, 9.0f, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT); // was 5, now 9
}

int main() {
  float *F;
  double *D;
  CHECK(hipMalloc(&F, 6 * sizeof(float)));
  CHECK(hipMalloc(&D, 2 * sizeof(double)));

  float FInit[6] = {1e30f, -1e30f, 10.0f, 0.0f, 0.0f, 0.0f};
  double DInit[2] = {1e300, -1e300};
  CHECK(hipMemcpy(F, FInit, sizeof(FInit), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(D, DInit, sizeof(DInit), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(reduceFloat, dim3(1), dim3(NumThreads), 0, 0, &F[0],
                     &F[1]);
  hipLaunchKernelGGL(reduceDouble, dim3(1), dim3(NumThreads), 0, 0, &D[0],
                     &D[1]);
  hipLaunchKernelGGL(returnsPreviousValue, dim3(1), dim3(1), 0, 0, &F[2],
                     &F[3]);
  CHECK(hipDeviceSynchronize());

  float FOut[6];
  double DOut[2];
  CHECK(hipMemcpy(FOut, F, sizeof(FOut), hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(DOut, D, sizeof(DOut), hipMemcpyDeviceToHost));

  int Errors = 0;
  if (FOut[0] != -100.0f || FOut[1] != float(NumThreads - 1) - 100.0f) {
    printf("float reduction: min %f max %f, expected %f %f\n", FOut[0], FOut[1],
           -100.0f, float(NumThreads - 1) - 100.0f);
    ++Errors;
  }
  if (DOut[0] != -100.0 || DOut[1] != double(NumThreads - 1) - 100.0) {
    printf("double reduction: min %f max %f, expected %f %f\n", DOut[0],
           DOut[1], -100.0, double(NumThreads - 1) - 100.0);
    ++Errors;
  }
  if (FOut[3] != 10.0f || FOut[4] != 5.0f || FOut[5] != 5.0f) {
    printf("returned previous values: %f %f %f, expected 10 5 5\n", FOut[3],
           FOut[4], FOut[5]);
    ++Errors;
  }
  if (FOut[2] != 9.0f) {
    printf("final cell %f, expected 9\n", FOut[2]);
    ++Errors;
  }

  CHECK(hipFree(F));
  CHECK(hipFree(D));

  if (Errors) {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
