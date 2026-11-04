// Reproduces #2352: graph memset node APIs accepted fills that do not fit the allocation.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

struct Case {
  size_t Off;
  size_t ES, W, H, P;
  bool Stack;
  bool Ok;
};

static hipMemsetParams mk(void *Dst, const Case &C) {
  hipMemsetParams P = {};
  P.dst = Dst;
  P.elementSize = C.ES;
  P.width = C.W;
  P.height = C.H;
  P.pitch = C.P;
  return P;
}

int main() {
  const size_t M = SIZE_MAX / 2;
  const Case Cases[] = {
      {0, 1, 128, 1, 128, false, false}, {0, 1, 8, 2, 1000000, false, false},
      {0, 1, 8, 2, 4, false, false},     {0, 4, 17, 1, 0, false, false},
      {0, 4, 2, 2, 4, false, false},     {0, 4, 8, 2, 40, false, false},
      {32, 1, 40, 1, 0, false, false},   {0, 1, 8, M, M, false, false},
      {0, 4, M, 1, 0, false, false},     {0, 1, 8, 1, 0, true, false},
      {0, 1, 64, 1, 0, false, true},     {0, 1, 16, 2, 32, false, true},
      {0, 1, 16, 3, 24, false, true},    {0, 4, 16, 1, 0, false, true},
      {0, 4, 4, 2, 32, false, true},     {32, 1, 32, 1, 999, false, true},
  };
  char *Base;
  char Stack[64];
  if (hipMalloc(&Base, 64) != hipSuccess)
    return 1;
  hipGraph_t G;
  hipGraphNode_t Good;
  hipGraphExec_t Exec;
  hipMemsetParams GP = mk(Base, {0, 1, 16, 1, 0, false, true});
  if (hipGraphCreate(&G, 0) != hipSuccess ||
      hipGraphAddMemsetNode(&Good, G, nullptr, 0, &GP) != hipSuccess ||
      hipGraphInstantiate(&Exec, G, nullptr, nullptr, 0) != hipSuccess)
    return 1;

  int Fails = 0;
  for (const Case &C : Cases) {
    hipMemsetParams P = mk(C.Stack ? (void *)Stack : (void *)(Base + C.Off), C);
    hipGraphNode_t N;
    hipError_t R[3] = {hipGraphAddMemsetNode(&N, G, nullptr, 0, &P),
                       hipGraphMemsetNodeSetParams(Good, &P),
                       hipGraphExecMemsetNodeSetParams(Exec, Good, &P)};
    const char *Api[3] = {"Add", "SetParams", "ExecSetParams"};
    for (int I = 0; I < 3; ++I)
      if ((R[I] == hipSuccess) != C.Ok) {
        printf("FAIL: %s es=%zu w=%zu h=%zu p=%zu off=%zu stack=%d returned %s, "
               "expected %s\n",
               Api[I], C.ES, C.W, C.H, C.P, C.Off, (int)C.Stack,
               hipGetErrorName(R[I]), C.Ok ? "hipSuccess" : "hipErrorInvalidValue");
        ++Fails;
      }
  }
  if (Fails)
    return 1;
  printf("PASS\n");
  return 0;
}
