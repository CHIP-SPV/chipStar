// Reproduces CHIP-SPV/chipStar#1750: __funnelshift_l, _lc and _rc return wrong
// values. Expected values follow the CUDA Math API definitions.

#include <hip/hip_runtime.h>
#include <cstdio>

constexpr unsigned N = 40;
constexpr unsigned Lo = 0x89abcdefu, Hi = 0xfedcba98u;

__global__ void run(unsigned *Out) {
  for (unsigned S = 0; S < N; ++S) {
    Out[4 * S + 0] = __funnelshift_l(Lo, Hi, S);
    Out[4 * S + 1] = __funnelshift_lc(Lo, Hi, S);
    Out[4 * S + 2] = __funnelshift_r(Lo, Hi, S);
    Out[4 * S + 3] = __funnelshift_rc(Lo, Hi, S);
  }
}

int main() {
  unsigned *Out;
  if (hipMalloc(&Out, 4 * N * sizeof(unsigned)) != hipSuccess)
    return 1;
  hipLaunchKernelGGL(run, dim3(1), dim3(1), 0, 0, Out);
  unsigned Got[4 * N];
  if (hipMemcpy(Got, Out, sizeof(Got), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  hipFree(Out);

  const char *Names[4] = {"l", "lc", "r", "rc"};
  unsigned long long Concat = (unsigned long long)Hi << 32 | Lo;
  int Errors = 0;
  for (unsigned S = 0; S < N; ++S) {
    unsigned C = S < 32 ? S : 32;
    unsigned Want[4] = {(unsigned)(Concat << (S & 31) >> 32),
                        (unsigned)(Concat << C >> 32),
                        (unsigned)(Concat >> (S & 31)),
                        (unsigned)(Concat >> C)};
    for (int F = 0; F < 4; ++F)
      if (Got[4 * S + F] != Want[F]) {
        printf("__funnelshift_%s(shift=%u): got 0x%08x want 0x%08x\n",
               Names[F], S, Got[4 * S + F], Want[F]);
        ++Errors;
      }
  }
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
