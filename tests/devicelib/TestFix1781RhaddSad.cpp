// Reproduces CHIP-SPV/chipStar#1781: __urhadd, __sad and __usad return wrong
// values. Expected values follow the CUDA Math API definitions.

#include <hip/hip_runtime.h>
#include <climits>
#include <cstdio>

struct Case {
  int Sa, Sb;
  unsigned Ua, Ub;
};
constexpr int N = 6;
constexpr unsigned Z = 7;
const Case Cases[N] = {{INT_MIN, INT_MAX, 0, 0xffffffffu},
                       {INT_MAX, INT_MIN, 0xffffffffu, 0},
                       {INT_MAX, INT_MAX, 0xffffffffu, 0xffffffffu},
                       {-1, 0, 1, 2},
                       {1, 2, 5, 3},
                       {-5, 3, 0x80000000u, 0x7fffffffu}};

__global__ void run(const Case *In, unsigned *Out) {
  for (int I = 0; I < N; ++I) {
    Out[4 * I + 0] = __rhadd(In[I].Sa, In[I].Sb);
    Out[4 * I + 1] = __urhadd(In[I].Ua, In[I].Ub);
    Out[4 * I + 2] = __sad(In[I].Sa, In[I].Sb, Z);
    Out[4 * I + 3] = __usad(In[I].Ua, In[I].Ub, Z);
  }
}

int main() {
  Case *In;
  unsigned *Out;
  if (hipMalloc(&In, sizeof(Cases)) != hipSuccess ||
      hipMalloc(&Out, 4 * N * sizeof(unsigned)) != hipSuccess ||
      hipMemcpy(In, Cases, sizeof(Cases), hipMemcpyHostToDevice) != hipSuccess)
    return 1;
  hipLaunchKernelGGL(run, dim3(1), dim3(1), 0, 0, In, Out);
  unsigned Got[4 * N];
  if (hipMemcpy(Got, Out, sizeof(Got), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  hipFree(In);
  hipFree(Out);

  const char *Names[4] = {"__rhadd", "__urhadd", "__sad", "__usad"};
  int Errors = 0;
  for (int I = 0; I < N; ++I) {
    const Case &C = Cases[I];
    long long S = (long long)C.Sa - C.Sb, U = (long long)C.Ua - C.Ub;
    unsigned Want[4] = {(unsigned)(int)(((long long)C.Sa + C.Sb + 1) >> 1),
                        (unsigned)(((long long)C.Ua + C.Ub + 1) >> 1),
                        (unsigned)(S < 0 ? -S : S) + Z,
                        (unsigned)(U < 0 ? -U : U) + Z};
    for (int F = 0; F < 4; ++F)
      if (Got[4 * I + F] != Want[F]) {
        printf("%s case %d: got 0x%08x want 0x%08x\n", Names[F], I,
               Got[4 * I + F], Want[F]);
        ++Errors;
      }
  }
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
