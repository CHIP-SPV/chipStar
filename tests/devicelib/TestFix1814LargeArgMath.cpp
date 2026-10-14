// Reproduces https://github.com/CHIP-SPV/chipStar/issues/1814: double
// j0/j1/y0/y1 are wrong for x >= 2^30 (generic trig_preop returns 0) and NaN
// from about 1e38 (OCML seeds double rsqrt with float native_rsqrt).

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

#define CHECK(X)                                                               \
  if ((X) != hipSuccess) {                                                     \
    printf("FAILED: %s\n", #X);                                                \
    return 1;                                                                  \
  }

constexpr int N = 5, F = 7;

__global__ void run(const double *In, double *Out) {
  double X = In[threadIdx.x];
  double *O = Out + F * threadIdx.x;
  O[0] = j0(X);
  O[1] = j1(X);
  O[2] = y0(X);
  O[3] = y1(X);
  O[4] = sin(X);
  O[5] = cos(X);
  O[6] = rsqrt(X);
}

int main() {
  const double In[N] = {1e9, 2e9, 1e12, 1e100, 1e300};
  double Out[N * F], *DevIn, *DevOut;
  CHECK(hipMalloc(&DevIn, sizeof(In)));
  CHECK(hipMalloc(&DevOut, sizeof(Out)));
  CHECK(hipMemcpy(DevIn, In, sizeof(In), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(run, dim3(1), dim3(N), 0, 0, DevIn, DevOut);
  CHECK(hipGetLastError());
  CHECK(hipMemcpy(Out, DevOut, sizeof(Out), hipMemcpyDeviceToHost));

  const char *Names[F] = {"j0", "j1", "y0", "y1", "sin", "cos", "rsqrt"};
  int Errors = 0;
  for (int I = 0; I < N; ++I) {
    double X = In[I];
    double Amp = std::sqrt(2 / (M_PI * X)); // Bessel envelope
    double Want[F] = {::j0(X),     ::j1(X),     ::y0(X),         ::y1(X),
                      std::sin(X), std::cos(X), 1 / std::sqrt(X)};
    double Scale[F] = {Amp, Amp, Amp, Amp, 1, 1, Want[6]};
    for (int K = 0; K < F; ++K) {
      double Got = Out[F * I + K];
      if (!(std::fabs(Got - Want[K]) <= 1e-6 * Scale[K])) { // also rejects NaN
        printf("%s(%g): got %.17g, expected %.17g\n", Names[K], In[I], Got,
               Want[K]);
        ++Errors;
      }
    }
  }
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
