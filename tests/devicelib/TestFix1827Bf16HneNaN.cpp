// Reproduces CHIP-SPV/chipStar#1827: bf16 __hne, __hne2 and __hbne2 are true
// for NaN. Expected values follow the CUDA Math API bfloat16 comparison
// definitions.

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

constexpr int NFn = 5, NCase = 5;
const char *Names[NFn] = {"__hne", "__hne2.x", "__hne2.y", "__hbne2",
                          "__hbne2 (other lane differs)"};
// Operands per case: (NaN, 1), (1, NaN), (1, 1), (1, 2), (2, 1).
const int Want[NCase] = {0, 0, 0, 1, 1};

__global__ void run(const float *A, const float *B, int *Out) {
  for (int C = 0; C < NCase; ++C) {
    __hip_bfloat16 X = __float2bfloat16(A[C]), Y = __float2bfloat16(B[C]);
    __hip_bfloat162 X2 = __halves2bfloat162(X, X), Y2 = __halves2bfloat162(Y, Y);
    __hip_bfloat162 Ne2 = __hne2(X2, Y2);
    int *O = Out + C * NFn;
    O[0] = __hne(X, Y);
    O[1] = __low2float(Ne2) == 1.0f ? 1 : __low2float(Ne2) == 0.0f ? 0 : -1;
    O[2] = __high2float(Ne2) == 1.0f ? 1 : __high2float(Ne2) == 0.0f ? 0 : -1;
    O[3] = __hbne2(X2, Y2);
    O[4] = __hbne2(__halves2bfloat162(X, __float2bfloat16(0.0f)),
                   __halves2bfloat162(Y, __float2bfloat16(2.0f)));
  }
}

int main() {
  float HA[NCase] = {std::nanf(""), 1.0f, 1.0f, 1.0f, 2.0f};
  float HB[NCase] = {1.0f, std::nanf(""), 1.0f, 2.0f, 1.0f};
  float *A, *B;
  int *Out, Got[NCase * NFn];
  if (hipMalloc(&A, sizeof(HA)) != hipSuccess ||
      hipMalloc(&B, sizeof(HB)) != hipSuccess ||
      hipMalloc(&Out, sizeof(Got)) != hipSuccess ||
      hipMemcpy(A, HA, sizeof(HA), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(B, HB, sizeof(HB), hipMemcpyHostToDevice) != hipSuccess)
    return 1;
  hipLaunchKernelGGL(run, dim3(1), dim3(1), 0, 0, A, B, Out);
  if (hipMemcpy(Got, Out, sizeof(Got), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;

  int Errors = 0;
  for (int C = 0; C < NCase; ++C)
    for (int F = 0; F < NFn; ++F)
      if (Got[C * NFn + F] != Want[C]) {
        printf("%s(%g, %g): got %d want %d\n", Names[F], HA[C], HB[C],
               Got[C * NFn + F], Want[C]);
        ++Errors;
      }
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
