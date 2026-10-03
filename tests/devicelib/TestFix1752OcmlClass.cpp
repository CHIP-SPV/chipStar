// Reproduces https://github.com/CHIP-SPV/chipStar/issues/1752: OCML's class
// test treats normal numbers as NaN, so cyl_bessel_i0f(2) returns 2.

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

#define CHECK(X)                                                               \
  if ((X) != hipSuccess) {                                                     \
    printf("FAILED: %s\n", #X);                                                \
    return 1;                                                                  \
  }

__global__ void run(const float *FIn, const double *DIn, float *FOut,
                    double *DOut, int *IOut) {
  FOut[0] = cyl_bessel_i0f(FIn[0]);
  FOut[1] = cyl_bessel_i1f(FIn[0]);
  DOut[0] = cyl_bessel_i0(DIn[0]);
  DOut[1] = cyl_bessel_i1(DIn[0]);
  IOut[0] = ::isfinite(FIn[0]);
}

int main() {
  const float FIn = 2.0f;
  const double DIn = 2.0;
  float *DevFIn, *DevFOut;
  double *DevDIn, *DevDOut;
  int *DevIOut;
  CHECK(hipMalloc(&DevFIn, sizeof(float)));
  CHECK(hipMalloc(&DevDIn, sizeof(double)));
  CHECK(hipMalloc(&DevFOut, 2 * sizeof(float)));
  CHECK(hipMalloc(&DevDOut, 2 * sizeof(double)));
  CHECK(hipMalloc(&DevIOut, sizeof(int)));
  CHECK(hipMemcpy(DevFIn, &FIn, sizeof(float), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(DevDIn, &DIn, sizeof(double), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(run, dim3(1), dim3(1), 0, 0, DevFIn, DevDIn, DevFOut,
                     DevDOut, DevIOut);
  CHECK(hipGetLastError());
  float F[2];
  double D[2];
  int I;
  CHECK(hipMemcpy(F, DevFOut, sizeof(F), hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(D, DevDOut, sizeof(D), hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(&I, DevIOut, sizeof(I), hipMemcpyDeviceToHost));

  // I0(2) and I1(2), modified Bessel functions of the first kind.
  const double I0 = 2.2795853023360673, I1 = 1.5906368546373291;
  int Errors = 0;
  auto check = [&](const char *Name, double Got, double Want, double Tol) {
    if (!(std::fabs(Got - Want) <= Tol * Want)) { // also rejects NaN
      printf("%s: got %.17g, expected %.17g\n", Name, Got, Want);
      ++Errors;
    }
  };
  check("cyl_bessel_i0f(2)", F[0], I0, 1e-5);
  check("cyl_bessel_i1f(2)", F[1], I1, 1e-5);
  check("cyl_bessel_i0(2)", D[0], I0, 1e-12);
  check("cyl_bessel_i1(2)", D[1], I1, 1e-12);
  if (!I) {
    printf("isfinite(2.0f): got 0\n");
    ++Errors;
  }
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
