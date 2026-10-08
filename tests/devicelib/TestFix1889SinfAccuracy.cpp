// Checks __sinf accuracy: https://github.com/CHIP-SPV/chipStar/issues/1889

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

#define CHECK(X)                                                               \
  if ((X) != hipSuccess) {                                                     \
    printf("FAILED: %s\n", #X);                                                \
    return 1;                                                                  \
  }

__global__ void run(const float *In, float *Out) {
  int I = blockIdx.x * blockDim.x + threadIdx.x;
  Out[I] = __sinf(In[I]);
}

int main() {
  constexpr int N = 1 << 16;
  static float In[N], Out[N];
  for (int I = 0; I < N; ++I)
    In[I] = (float)(-M_PI + 2 * M_PI * I / (N - 1));
  float *DevIn, *DevOut;
  CHECK(hipMalloc(&DevIn, sizeof(In)));
  CHECK(hipMalloc(&DevOut, sizeof(Out)));
  CHECK(hipMemcpy(DevIn, In, sizeof(In), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(run, dim3(N / 256), dim3(256), 0, 0, DevIn, DevOut);
  CHECK(hipGetLastError());
  CHECK(hipMemcpy(Out, DevOut, sizeof(Out), hipMemcpyDeviceToHost));

  // CUDA Programming Guide: __sinf max abs error on [-pi, pi] is 2^-21.41.
  double MaxErr = 0;
  int Errors = 0;
  for (int I = 0; I < N; ++I) {
    double Err = fabs(Out[I] - sin((double)In[I]));
    MaxErr = fmax(MaxErr, Err);
    Errors += !(Err <= exp2(-21.41)); // NaN counts as an error
  }
  printf("__sinf max abs error on [-pi, pi]: %.3e\n", MaxErr);
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
