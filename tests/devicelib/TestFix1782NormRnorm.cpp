// Reproduces https://github.com/CHIP-SPV/chipStar/issues/1782: norm(double)
// accumulates in float, and rnormf/rnorm return the norm, not its reciprocal.

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

#define CHECK(X)                                                               \
  if ((X) != hipSuccess) {                                                     \
    printf("FAILED: %s\n", #X);                                                \
    return 1;                                                                  \
  }

// {1, 1e-4}: the 1e-8 term vanishes in a float sum; {1e30, 1e30}: overflows it.
constexpr int NV = 4, Dim = 3;
const double In[NV][Dim] = {
    {1, 1, 0}, {3, 4, 12}, {1, 1e-4, 0}, {1e30, 1e30, 0}};

// Separate kernels: under IGC fp64 emulation a kernel mixing both hangs the GPU.
__global__ void runFloat(const float *F, float *FOut) {
  for (int V = 0; V < NV; ++V) {
    FOut[2 * V] = normf(Dim, F + V * Dim);
    FOut[2 * V + 1] = rnormf(Dim, F + V * Dim);
  }
}

__global__ void runDouble(const double *D, double *DOut) {
  for (int V = 0; V < NV; ++V) {
    DOut[2 * V] = norm(Dim, D + V * Dim);
    DOut[2 * V + 1] = rnorm(Dim, D + V * Dim);
  }
}

int Errors = 0;
void check(const char *Name, int V, long double Got, long double Want,
           long double Tol) {
  if (!(fabsl(Got - Want) <= Tol * fabsl(Want))) {
    printf("%s(vector %d): got %.17Lg want %.17Lg\n", Name, V, Got, Want);
    ++Errors;
  }
}

int main() {
  float F[NV * Dim];
  double D[NV * Dim];
  for (int I = 0; I < NV * Dim; ++I)
    F[I] = D[I] = In[I / Dim][I % Dim];
  float *DevF, *DevFOut;
  double *DevD, *DevDOut;
  CHECK(hipMalloc(&DevF, sizeof(F)));
  CHECK(hipMalloc(&DevD, sizeof(D)));
  CHECK(hipMalloc(&DevFOut, 2 * NV * sizeof(float)));
  CHECK(hipMalloc(&DevDOut, 2 * NV * sizeof(double)));
  CHECK(hipMemcpy(DevF, F, sizeof(F), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(DevD, D, sizeof(D), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(runFloat, dim3(1), dim3(1), 0, 0, DevF, DevFOut);
  hipLaunchKernelGGL(runDouble, dim3(1), dim3(1), 0, 0, DevD, DevDOut);
  CHECK(hipGetLastError());
  float FOut[2 * NV];
  double DOut[2 * NV];
  CHECK(hipMemcpy(FOut, DevFOut, sizeof(FOut), hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(DOut, DevDOut, sizeof(DOut), hipMemcpyDeviceToHost));
  hipDeviceProp_t Props;
  CHECK(hipGetDeviceProperties(&Props, 0));

  for (int V = 0; V < NV; ++V) {
    long double SF = 0, SD = 0;
    for (int I = 0; I < Dim; ++I) {
      SF += (long double)F[V * Dim + I] * F[V * Dim + I];
      SD += (long double)D[V * Dim + I] * D[V * Dim + I];
    }
    // {1e30, 1e30} is out of float range: double only.
    if (V != 3) {
      check("normf", V, FOut[2 * V], sqrtl(SF), 1e-6L);
      check("rnormf", V, FOut[2 * V + 1], 1 / sqrtl(SF), 1e-6L);
    }
    // Without cl_khr_fp64 (rusticl) double math has no accuracy contract.
    if (Props.arch.hasDoubles) {
      check("norm", V, DOut[2 * V], sqrtl(SD), 1e-14L);
      check("rnorm", V, DOut[2 * V + 1], 1 / sqrtl(SD), 1e-14L);
    }
  }
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
