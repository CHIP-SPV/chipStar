// Reproduces CHIP-SPV/chipStar#1828: __half <=, >= are true for NaN, and
// __half2 <=, >=, != need only one lane instead of both. Expected values follow
// CUDA's operators (__hle, __hge, __hble2, __hbge2, __hbneu2).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

constexpr int NCase = 18;
const char *Names[NCase] = {
    "NaN<=1",          "NaN>=1",          "1<=NaN",
    "1>=NaN",          "1<=2",            "1>=2",
    "1<=1",            "1>=1",            "(1,5)<=(2,3)",
    "(2,3)>=(1,5)",    "(1,2)<=(1,2)",    "(1,2)>=(1,2)",
    "(NaN,NaN)<=(1,1)", "(NaN,NaN)>=(1,1)", "(1,2)!=(1,3)",
    "(1,2)!=(3,4)",    "(NaN,2)!=(1,2)",  "(NaN,NaN)!=(1,1)"};
const int Want[NCase] = {0, 0, 0, 0, 1, 0, 1, 1, 0,
                         0, 1, 1, 0, 0, 0, 1, 0, 1};

__host__ __device__ void eval(float N, int *O) {
  __half H1 = 1.0f, H2 = 2.0f, HN = N;
  __half2 A{1.0f, 5.0f}, B{2.0f, 3.0f}, C{1.0f, 2.0f}, D{1.0f, 3.0f},
      E{3.0f, 4.0f}, F{N, 2.0f}, G{N, N}, I{1.0f, 1.0f};
  O[0] = HN <= H1;  O[1] = HN >= H1;  O[2] = H1 <= HN;  O[3] = H1 >= HN;
  O[4] = H1 <= H2;  O[5] = H1 >= H2;  O[6] = H1 <= H1;  O[7] = H1 >= H1;
  O[8] = A <= B;    O[9] = B >= A;    O[10] = C <= C;   O[11] = C >= C;
  O[12] = G <= I;   O[13] = G >= I;   O[14] = C != D;   O[15] = C != E;
  O[16] = F != C;   O[17] = G != I;
}

__global__ void run(float N, int *O) { eval(N, O); }

int check(const char *Where, const int *Got) {
  int Errors = 0;
  for (int C = 0; C < NCase; ++C)
    if (Got[C] != Want[C]) {
      printf("%s %s: got %d want %d\n", Where, Names[C], Got[C], Want[C]);
      ++Errors;
    }
  return Errors;
}

int main() {
  int Host[NCase], Dev[NCase], *Out;
  eval(std::nanf(""), Host);
  if (hipMalloc(&Out, sizeof(Dev)) != hipSuccess)
    return 1;
  hipLaunchKernelGGL(run, dim3(1), dim3(1), 0, 0, std::nanf(""), Out);
  if (hipMemcpy(Dev, Out, sizeof(Dev), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  int Errors = check("host", Host) + check("device", Dev);
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
