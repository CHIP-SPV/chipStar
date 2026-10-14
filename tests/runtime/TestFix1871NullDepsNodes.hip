// Reproduces #1871: hipGraphAddEmptyNode and hipGraphAddMemcpyNode{From,To}Symbol
// dereference a null pDependencies with numDependencies != 0 (and the symbol
// variants a null pGraphNode or graph) instead of returning hipErrorInvalidValue.
// The name avoids "graph": tests/known_failures.yaml masks '.*[Gg]raph.*'.
#include <hip/hip_runtime.h>
#include <cstdio>

__device__ int Sym;

#define EXPECT_INVALID(x)                                                      \
  do {                                                                         \
    hipError_t Err = (x);                                                      \
    if (Err != hipErrorInvalidValue) {                                         \
      printf("FAIL: %s returned %d (%s)\n", #x, Err, hipGetErrorName(Err));    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  hipGraph_t G;
  if (hipGraphCreate(&G, 0) != hipSuccess)
    return 1;
  hipGraphNode_t N;
  int H = 0;
  const void *S = &Sym;
  EXPECT_INVALID(hipGraphAddEmptyNode(&N, G, nullptr, 1));
  EXPECT_INVALID(hipGraphAddMemcpyNodeFromSymbol(&N, G, nullptr, 1, &H, S,
                                                 sizeof(int), 0,
                                                 hipMemcpyDeviceToHost));
  EXPECT_INVALID(hipGraphAddMemcpyNodeToSymbol(&N, G, nullptr, 1, S, &H,
                                               sizeof(int), 0,
                                               hipMemcpyHostToDevice));
  EXPECT_INVALID(hipGraphAddMemcpyNodeFromSymbol(nullptr, G, nullptr, 0, &H, S,
                                                 sizeof(int), 0,
                                                 hipMemcpyDeviceToHost));
  EXPECT_INVALID(hipGraphAddMemcpyNodeFromSymbol(&N, nullptr, nullptr, 0, &H, S,
                                                 sizeof(int), 0,
                                                 hipMemcpyDeviceToHost));
  EXPECT_INVALID(hipGraphAddMemcpyNodeToSymbol(nullptr, G, nullptr, 0, S, &H,
                                               sizeof(int), 0,
                                               hipMemcpyHostToDevice));
  EXPECT_INVALID(hipGraphAddMemcpyNodeToSymbol(&N, nullptr, nullptr, 0, S, &H,
                                               sizeof(int), 0,
                                               hipMemcpyHostToDevice));
  (void)hipGraphDestroy(G);
  printf("PASSED\n");
  return 0;
}
