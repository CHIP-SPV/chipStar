// Reproduces CHIP-SPV/chipStar#1813: __hne, __hne2 and __hbne2 are true for
// NaN, and __hb{eq,le,ge,lt,gt}u2 are false for NaN. Expected values follow the
// CUDA Math API half/half2 comparison definitions.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

constexpr int NFn = 10, NCase = 4;
const char *Names[NFn] = {"__hne",    "__hne2.x", "__hne2.y", "__hbne2",
                          "__hbequ2", "__hbneu2", "__hbleu2", "__hbgeu2",
                          "__hbltu2", "__hbgtu2"};
// Operands per case: (NaN, 1), (1, NaN), (1, 1), (1, 2).
const int Want[NFn][NCase] = {{0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1},
                              {0, 0, 0, 1}, {1, 1, 1, 0}, {1, 1, 0, 1},
                              {1, 1, 1, 1}, {1, 1, 1, 0}, {1, 1, 0, 1},
                              {1, 1, 0, 0}};

__global__ void run(const float *A, const float *B, int *Out) {
  for (int C = 0; C < NCase; ++C) {
    __half X = __float2half(A[C]), Y = __float2half(B[C]);
    __half2 X2 = __halves2half2(X, X), Y2 = __halves2half2(Y, Y);
    __half2 Ne2 = __hne2(X2, Y2);
    int *O = Out + C * NFn;
    O[0] = __hne(X, Y);
    O[1] = __low2float(Ne2) == 1.0f ? 1 : __low2float(Ne2) == 0.0f ? 0 : -1;
    O[2] = __high2float(Ne2) == 1.0f ? 1 : __high2float(Ne2) == 0.0f ? 0 : -1;
    O[3] = __hbne2(X2, Y2);
    O[4] = __hbequ2(X2, Y2);
    O[5] = __hbneu2(X2, Y2);
    O[6] = __hbleu2(X2, Y2);
    O[7] = __hbgeu2(X2, Y2);
    O[8] = __hbltu2(X2, Y2);
    O[9] = __hbgtu2(X2, Y2);
  }
}

int main() {
  float HA[NCase] = {std::nanf(""), 1.0f, 1.0f, 1.0f};
  float HB[NCase] = {1.0f, std::nanf(""), 1.0f, 2.0f};
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
      if (Got[C * NFn + F] != Want[F][C]) {
        printf("%s(%g, %g): got %d want %d\n", Names[F], HA[C], HB[C],
               Got[C * NFn + F], Want[F][C]);
        ++Errors;
      }
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
