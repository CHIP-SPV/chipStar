// Reproduces https://github.com/CHIP-SPV/chipStar/issues/2018: cyl_bessel_i1(x) is NaN for x <= -8.

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

#define CHECK(X)                                                               \
  if ((X) != hipSuccess) {                                                     \
    printf("FAILED: %s\n", #X);                                                \
    return 1;                                                                  \
  }

// 710 exercises the x > 709 branch.
constexpr int N = 3;
const double In[N] = {8, 20, 710};

__global__ void run(const double *X, double *Out) {
  for (int I = 0; I < N; ++I) {
    Out[2 * I] = cyl_bessel_i1(X[I]);
    Out[2 * I + 1] = cyl_bessel_i1(-X[I]);
  }
}

int main() {
  hipDeviceProp_t Props;
  CHECK(hipGetDeviceProperties(&Props, 0));
  if (!Props.arch.hasDoubles) {
    printf("HIP_SKIP_THIS_TEST: device has no fp64\n");
    return 0;
  }
  double *DevX, *DevOut;
  CHECK(hipMalloc(&DevX, sizeof(In)));
  CHECK(hipMalloc(&DevOut, 2 * N * sizeof(double)));
  CHECK(hipMemcpy(DevX, In, sizeof(In), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(run, dim3(1), dim3(1), 0, 0, DevX, DevOut);
  CHECK(hipGetLastError());
  double Out[2 * N];
  CHECK(hipMemcpy(Out, DevOut, sizeof(Out), hipMemcpyDeviceToHost));

  int Errors = 0;
  for (int I = 0; I < N; ++I) {
    double P = Out[2 * I], M = Out[2 * I + 1];
    if (!std::isfinite(P) || !std::isfinite(M) || M != -P) {
      printf("cyl_bessel_i1(%g) = %g, cyl_bessel_i1(%g) = %g\n", In[I], P,
             -In[I], M);
      ++Errors;
    }
  }
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
