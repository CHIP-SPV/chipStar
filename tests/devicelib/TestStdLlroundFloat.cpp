// Reproduces: the type-crossing rounding intrinsics (floating point in,
// integer out) have no single OpenCL.std ExtInst, because every OpenCL.std
// rounding instruction is float to float. A producer that does not expand them
// itself rejects the module outright:
//
//   InvalidFunctionCall: Unexpected llvm intrinsic: llvm.llround.i64.f32
//   clang++: error: hipspv-link command failed with exit code 8
//
// and with the in-tree SPIR-V backend:
//
//   LLVM ERROR: unable to legalize instruction: %8:iid(s64) = G_INTRINSIC_LRINT
//
// The failure surfaces at hipspv-link with no source location at all.
//
// libstdc++'s float overloads are constexpr, so HIP has already made them
// implicitly __host__ __device__ and they win overload resolution over
// chipStar's own device declarations. That is how std::llround and std::llrint
// on a float reach llvm.llround / llvm.llrint, and how std::lround on a float
// reaches llvm.lround. std::lrint is different: chipStar declares lrint(float)
// in sp_math.hh and pulls it into namespace std, so std::lrint(float) resolves
// to the devicelib entry point and never forms the intrinsic. llvm.lrint is
// therefore reached here through __builtin_lrintf directly.

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

__global__ void run(const float *In, const double *InD, long long *Out,
                    long *OutL) {
  using std::llrint;
  using std::llround;
  using std::lround;
  Out[0] = llround(In[0]);
  Out[1] = llround(In[1]);
  Out[2] = llrint(In[0]);
  Out[3] = llround(InD[0]);
  Out[4] = llrint(InD[0]);

  // std::lrint(float) resolves to chipStar's devicelib overload rather than
  // libstdc++'s, so reach llvm.lrint through the builtin instead.
  OutL[0] = __builtin_lrintf(In[0]);
  OutL[1] = __builtin_lrintf(In[2]);
  OutL[2] = __builtin_lrint(InD[0]);
  OutL[3] = lround(In[0]);
  OutL[4] = lround(In[1]);
}

int main() {
  float HostIn[3] = {2.5f, -2.5f, 3.5f};
  double HostInD[1] = {7.5};

  float *In;
  double *InD;
  long long *Out;
  long *OutL;
  CHECK(hipMalloc(&In, sizeof(HostIn)));
  CHECK(hipMalloc(&InD, sizeof(HostInD)));
  CHECK(hipMalloc(&Out, 5 * sizeof(long long)));
  CHECK(hipMalloc(&OutL, 5 * sizeof(long)));
  CHECK(hipMemcpy(In, HostIn, sizeof(HostIn), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(InD, HostInD, sizeof(HostInD), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(run, dim3(1), dim3(1), 0, 0, In, InD, Out, OutL);
  CHECK(hipDeviceSynchronize());

  long long HostOut[5];
  long HostOutL[5];
  CHECK(hipMemcpy(HostOut, Out, sizeof(HostOut), hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(HostOutL, OutL, sizeof(HostOutL), hipMemcpyDeviceToHost));

  // llround and lround round half away from zero; llrint and lrint follow the
  // rounding mode, which is round to nearest even by default.
  const long long Expected[5] = {3, -3, 2, 8, 8};
  const char *Names[5] = {"llround(2.5f)", "llround(-2.5f)", "llrint(2.5f)",
                          "llround(7.5)", "llrint(7.5)"};
  const long ExpectedL[5] = {2, 4, 8, 3, -3};
  const char *NamesL[5] = {"lrint(2.5f)", "lrint(3.5f)", "lrint(7.5)",
                           "lround(2.5f)", "lround(-2.5f)"};

  int Errors = 0;
  for (int I = 0; I < 5; ++I)
    if (HostOut[I] != Expected[I]) {
      printf("%s: got %lld expected %lld\n", Names[I], HostOut[I], Expected[I]);
      ++Errors;
    }
  for (int I = 0; I < 5; ++I)
    if (HostOutL[I] != ExpectedL[I]) {
      printf("%s: got %ld expected %ld\n", NamesL[I], HostOutL[I],
             ExpectedL[I]);
      ++Errors;
    }

  CHECK(hipFree(In));
  CHECK(hipFree(InD));
  CHECK(hipFree(Out));
  CHECK(hipFree(OutL));

  if (Errors) {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
