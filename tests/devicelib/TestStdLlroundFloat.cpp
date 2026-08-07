// Reproduces: std::llround and std::llrint on a float expand to
// __builtin_llroundf / __builtin_llrintf, i.e. llvm.llround / llvm.llrint,
// which llvm-spirv cannot translate:
//
//   InvalidFunctionCall: Unexpected llvm intrinsic: llvm.llround.i64.f32
//   clang++: error: hipspv-link command failed with exit code 8
//
// libstdc++'s float overloads are constexpr, so HIP has already made them
// implicitly __host__ __device__ and they win overload resolution over
// chipStar's own device declarations. lround, lrint, nearbyint, rint, round
// and trunc all translate; only these two do not.

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

__global__ void run(const float *In, const double *InD, long long *Out) {
  using std::llrint;
  using std::llround;
  Out[0] = llround(In[0]);
  Out[1] = llround(In[1]);
  Out[2] = llrint(In[0]);
  Out[3] = llround(InD[0]);
  Out[4] = llrint(InD[0]);
}

int main() {
  float HostIn[2] = {2.5f, -2.5f};
  double HostInD[1] = {7.5};

  float *In;
  double *InD;
  long long *Out;
  CHECK(hipMalloc(&In, sizeof(HostIn)));
  CHECK(hipMalloc(&InD, sizeof(HostInD)));
  CHECK(hipMalloc(&Out, 5 * sizeof(long long)));
  CHECK(hipMemcpy(In, HostIn, sizeof(HostIn), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(InD, HostInD, sizeof(HostInD), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(run, dim3(1), dim3(1), 0, 0, In, InD, Out);
  CHECK(hipDeviceSynchronize());

  long long HostOut[5];
  CHECK(hipMemcpy(HostOut, Out, sizeof(HostOut), hipMemcpyDeviceToHost));

  // llround rounds half away from zero; llrint follows the rounding mode,
  // which is round to nearest even by default.
  const long long Expected[5] = {3, -3, 2, 8, 8};
  const char *Names[5] = {"llround(2.5f)", "llround(-2.5f)", "llrint(2.5f)",
                          "llround(7.5)", "llrint(7.5)"};

  int Errors = 0;
  for (int I = 0; I < 5; ++I)
    if (HostOut[I] != Expected[I]) {
      printf("%s: got %lld expected %lld\n", Names[I], HostOut[I], Expected[I]);
      ++Errors;
    }

  CHECK(hipFree(In));
  CHECK(hipFree(InD));
  CHECK(hipFree(Out));

  if (Errors) {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
