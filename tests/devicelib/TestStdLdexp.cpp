// Reproduces: std::ldexp in device code emits llvm.ldexp, which llvm-spirv
// cannot translate.
//
// libstdc++ declares std::ldexp as constexpr, which HIP turns into an
// implicitly __host__ __device__ function, so it wins overload resolution over
// chipStar's __device__ ldexp and expands to __builtin_ldexp. The module then
// carries an llvm.ldexp intrinsic and hipspv-link dies with
//
//   InvalidFunctionCall: Unexpected llvm intrinsic: llvm.ldexp.f64.i32
//
// with no source location at all. Same shape as the llvm.llround failure this
// pass was originally written for, and found the same way: it was sitting right
// behind it in Kokkos_CoreUnitTest_HIP.
//
// The exponents below are deliberately not compile time constants: a constant
// one is folded away before the pass ever sees an intrinsic, which would make
// the test pass even with the lowering removed.

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

#define CHECK(X)                                                               \
  do {                                                                         \
    hipError_t E = (X);                                                        \
    if (E != hipSuccess) {                                                     \
      printf("FAILED: %s -> %s\n", #X, hipGetErrorString(E));                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__device__ double callLdexpDouble(double X, int Exp) {
  using std::ldexp;
  return ldexp(X, Exp);
}

__device__ float callLdexpFloat(float X, int Exp) {
  using std::ldexp;
  return ldexp(X, Exp);
}

__global__ void run(double *DOut, float *FOut, const int *Exps) {
  DOut[0] = callLdexpDouble(3.0, Exps[0]);   //  3 * 2^4    = 48
  DOut[1] = callLdexpDouble(-1.5, Exps[1]);  // -1.5 * 2^-3 = -0.1875
  DOut[2] = callLdexpDouble(1.0, Exps[2]);   //  1 * 2^0    = 1
  DOut[3] = callLdexpDouble(0.0, Exps[0]);   //  0 stays 0
  FOut[0] = callLdexpFloat(3.0f, Exps[0]);
  FOut[1] = callLdexpFloat(-1.5f, Exps[1]);
  FOut[2] = callLdexpFloat(1.0f, Exps[2]);
  // A power of two large enough that a naive x * 2^n lowering would overflow
  // the intermediate: 2^-120 * 2^120 must come back as exactly 1.
  FOut[3] = callLdexpFloat(7.52316385e-37f /* 2^-120 */, Exps[3]);
}

int main() {
  const int HostExps[4] = {4, -3, 0, 120};
  int *Exps;
  double *DOut;
  float *FOut;
  CHECK(hipMalloc(&Exps, sizeof(HostExps)));
  CHECK(hipMalloc(&DOut, 4 * sizeof(double)));
  CHECK(hipMalloc(&FOut, 4 * sizeof(float)));
  CHECK(hipMemcpy(Exps, HostExps, sizeof(HostExps), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(run, dim3(1), dim3(1), 0, 0, DOut, FOut, Exps);
  CHECK(hipDeviceSynchronize());

  double D[4];
  float F[4];
  CHECK(hipMemcpy(D, DOut, sizeof(D), hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(F, FOut, sizeof(F), hipMemcpyDeviceToHost));

  const double DExpected[4] = {48.0, -0.1875, 1.0, 0.0};
  const float FExpected[4] = {48.0f, -0.1875f, 1.0f, 1.0f};

  int Errors = 0;
  for (int I = 0; I < 4; ++I) {
    if (D[I] != DExpected[I]) {
      printf("double ldexp #%d: got %g, expected %g\n", I, D[I], DExpected[I]);
      ++Errors;
    }
    if (F[I] != FExpected[I]) {
      printf("float ldexp #%d: got %g, expected %g\n", I, (double)F[I],
             (double)FExpected[I]);
      ++Errors;
    }
  }

  CHECK(hipFree(Exps));
  CHECK(hipFree(DOut));
  CHECK(hipFree(FOut));

  if (Errors) {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
