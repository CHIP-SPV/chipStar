// Reproduces: std::modf has no single-precision device overload.
//
// chipStar declares `__device__ float modf(float, float *)` in the global
// namespace but never pulls it into namespace std, so a device call written
// the portable way
//
//   using std::modf; modf(x, &integral);
//
// sees only libstdc++'s host-only float overload and fails to compile.
// Every other float function Kokkos routes through this idiom (frexp, ldexp,
// scalbn, nearbyint, lrint) survives because the double overload is a viable
// device candidate after a float to double conversion; modf cannot convert
// float * to double *, so it has no device candidate at all.

#include <hip/hip_runtime.h>
// <math.h> is deliberately included as well: libstdc++'s <math.h> re-exports
// std::modf into the global namespace, so declaring a *new* std::modf(float,
// float *) here instead of re-exporting the global one makes this line fail
// with "target of using declaration conflicts with declaration already in
// scope" and takes out every translation unit that includes <math.h>.
#include <math.h>
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

__device__ float callModfFloat(float X, float *Integral) {
  using std::modf;
  return modf(X, Integral);
}

__device__ double callModfDouble(double X, double *Integral) {
  using std::modf;
  return modf(X, Integral);
}

__global__ void run(float *FOut, double *DOut) {
  FOut[1] = callModfFloat(-3.25f, &FOut[0]);
  FOut[3] = callModfFloat(3.75f, &FOut[2]);
  DOut[1] = callModfDouble(-3.25, &DOut[0]);
}

int main() {
  float *FOut;
  double *DOut;
  CHECK(hipMalloc(&FOut, 4 * sizeof(float)));
  CHECK(hipMalloc(&DOut, 2 * sizeof(double)));

  hipLaunchKernelGGL(run, dim3(1), dim3(1), 0, 0, FOut, DOut);
  CHECK(hipDeviceSynchronize());

  float F[4];
  double D[2];
  CHECK(hipMemcpy(F, FOut, sizeof(F), hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(D, DOut, sizeof(D), hipMemcpyDeviceToHost));

  int Errors = 0;
  if (F[0] != -3.0f || F[1] != -0.25f) {
    printf("float modf(-3.25): integral %f fractional %f, expected -3 -0.25\n",
           F[0], F[1]);
    ++Errors;
  }
  if (F[2] != 3.0f || F[3] != 0.75f) {
    printf("float modf(3.75): integral %f fractional %f, expected 3 0.75\n",
           F[2], F[3]);
    ++Errors;
  }
  if (D[0] != -3.0 || D[1] != -0.25) {
    printf("double modf(-3.25): integral %f fractional %f, expected -3 -0.25\n",
           D[0], D[1]);
    ++Errors;
  }

  CHECK(hipFree(FOut));
  CHECK(hipFree(DOut));

  if (Errors) {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
