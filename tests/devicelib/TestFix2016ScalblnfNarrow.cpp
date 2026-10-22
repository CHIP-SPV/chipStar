// Reproduces https://github.com/CHIP-SPV/chipStar/issues/2016: scalblnf
// narrows a large negative long to int and wraps it positive, returning inf.

#include <hip/hip_runtime.h>
#include <climits>
#include <cmath>
#include <cstdio>

#define CHECK(X)                                                               \
  if ((X) != hipSuccess) {                                                     \
    printf("FAILED: %s\n", #X);                                                \
    return 1;                                                                  \
  }

__global__ void run(const long *In, float *Out) {
  Out[0] = scalblnf(1.0f, In[0]);
  Out[1] = scalblnf(1.0f, In[1]);
  Out[2] = scalblnf(1.0f, In[2]);
  Out[3] = scalblnf(3.0f, In[3]);
}

int main() {
  const long In[4] = {LONG_MIN, (long)INT_MIN - 1, LONG_MAX, 2L};
  const float Want[4] = {0.0f, 0.0f, INFINITY, 12.0f};
  long *DevIn;
  float *DevOut;
  CHECK(hipMalloc(&DevIn, sizeof(In)));
  CHECK(hipMalloc(&DevOut, sizeof(Want)));
  CHECK(hipMemcpy(DevIn, In, sizeof(In), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(run, dim3(1), dim3(1), 0, 0, DevIn, DevOut);
  CHECK(hipGetLastError());
  float Out[4];
  CHECK(hipMemcpy(Out, DevOut, sizeof(Out), hipMemcpyDeviceToHost));

  int Errors = 0;
  for (int I = 0; I < 4; ++I)
    if (Out[I] != Want[I]) {
      printf("scalblnf case %d (n=%ld): got %g, expected %g\n", I, In[I],
             Out[I], Want[I]);
      ++Errors;
    }
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
