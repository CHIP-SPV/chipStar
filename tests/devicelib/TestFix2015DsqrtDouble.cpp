// Checks __dsqrt_* compute in double precision:
// https://github.com/CHIP-SPV/chipStar/issues/2015

#include <hip/hip_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>

#define CHECK(X)                                                               \
  if ((X) != hipSuccess) {                                                     \
    printf("FAILED: %s\n", #X);                                                \
    return 1;                                                                  \
  }

__global__ void run(const double *In, double *Out) {
  int I = threadIdx.x;
  Out[4 * I + 0] = __dsqrt_rn(In[I]);
  Out[4 * I + 1] = __dsqrt_rd(In[I]);
  Out[4 * I + 2] = __dsqrt_ru(In[I]);
  Out[4 * I + 3] = __dsqrt_rz(In[I]);
}

int main() {
  const double In[3] = {2.0, 1e10, 1.0 + exp2(-40)};
  double Out[12];
  double *DevIn, *DevOut;
  CHECK(hipMalloc(&DevIn, sizeof(In)));
  CHECK(hipMalloc(&DevOut, sizeof(Out)));
  CHECK(hipMemcpy(DevIn, In, sizeof(In), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(run, dim3(1), dim3(3), 0, 0, DevIn, DevOut);
  CHECK(hipGetLastError());
  CHECK(hipMemcpy(Out, DevOut, sizeof(Out), hipMemcpyDeviceToHost));

  int Errors = 0;
  for (int I = 0; I < 12; ++I) {
    double Ref = sqrt(In[I / 4]);
    Errors += !(fabs(Out[I] - Ref) <= 4 * DBL_EPSILON * Ref); // NaN fails
    printf("%d: %.17g (expected %.17g)\n", I, Out[I], Ref);
  }
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
