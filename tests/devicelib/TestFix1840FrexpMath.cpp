// Reproduces https://github.com/CHIP-SPV/chipStar/issues/1840: OCML's generic
// BUILTIN_FREXP_EXP_F64 rounds to float and BUILTIN_FREXP_MANT_* return x,
// breaking double rhypot, norm3d/4d, rnorm3d/4d, rcbrt and small-x y0/y1/yn.

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

#define CHECK(X)                                                               \
  if ((X) != hipSuccess) {                                                     \
    printf("FAILED: %s\n", #X);                                                \
    return 1;                                                                  \
  }

constexpr int N = 20;

__global__ void run(double *O) {
  O[0] = rhypot(1e300, 1e300);
  O[1] = rhypot(1e-300, 1e-300);
  O[2] = norm3d(1e200, 1e200, 1e200);
  O[3] = rnorm3d(1e200, 1e200, 1e200);
  O[4] = norm4d(1e-200, 1e-200, 1e-200, 1e-200);
  O[5] = rnorm4d(1e200, 1e200, 1e200, 1e200);
  O[6] = rcbrt(1e300);
  O[7] = rcbrt(1e-300);
  O[8] = y0(0.1);
  O[9] = y0(0.2);
  O[10] = y1(0.1);
  O[11] = y1(0.2);
  O[12] = yn(2, 0.1);
  O[13] = rhypot(3.0, 4.0);
  O[14] = rcbrt(7.0);
  O[15] = y0(0.5);
  O[16] = rhypotf(3.0f, 4.0f);
  O[17] = rcbrtf(7.0f);
  O[18] = y0f(0.1f);
  O[19] = norm3df(1e20f, 1e20f, 1e20f);
}

int main() {
  double Out[N], *DevOut;
  CHECK(hipMalloc(&DevOut, sizeof(Out)));
  hipLaunchKernelGGL(run, dim3(1), dim3(1), 0, 0, DevOut);
  CHECK(hipGetLastError());
  CHECK(hipMemcpy(Out, DevOut, sizeof(Out), hipMemcpyDeviceToHost));

  const double Want[N] = {1 / std::hypot(1e300, 1e300),
                          1 / std::hypot(1e-300, 1e-300),
                          std::sqrt(3.0) * 1e200,
                          1 / (std::sqrt(3.0) * 1e200),
                          2e-200,
                          0.5e-200,
                          1e-100,
                          1e100,
                          ::y0(0.1),
                          ::y0(0.2),
                          ::y1(0.1),
                          ::y1(0.2),
                          ::yn(2, 0.1),
                          0.2,
                          1 / std::cbrt(7.0),
                          ::y0(0.5),
                          0.2,
                          1 / std::cbrt(7.0),
                          ::y0(0.1),
                          std::sqrt(3.0) * 1e20};
  int Errors = 0;
  for (int I = 0; I < N; ++I) {
    double Tol = I < 16 ? 1e-12 : 1e-5; // last four are float controls
    if (!(std::fabs(Out[I] - Want[I]) <= Tol * std::fabs(Want[I]))) {
      printf("result %d: got %.17g, expected %.17g\n", I, Out[I], Want[I]);
      ++Errors;
    }
  }
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
