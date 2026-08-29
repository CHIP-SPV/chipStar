// Reproduces #1524: hipGraphGetEdges ignores the caller's *numEdges capacity
// and writes every edge of the graph into from/to, overflowing arrays that are
// smaller than the edge count (Unit_hipGraphGetEdges_Functionality then aborts
// with "double free or corruption" when it frees them). It also never writes
// the actual edge count back, leaves the surplus entries untouched instead of
// null, returns every edge with from and to swapped, and accepts a null from
// with a non-null to.
//
// The test name deliberately avoids the word "graph": the '.*[Gg]raph.*' mask
// in tests/known_failures.yaml would otherwise exclude it from check.py runs.
#include <hip/hip_runtime_api.h>
#include <cstdio>

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t Err = (x);                                                      \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %d (%s)\n", #x, Err, hipGetErrorName(Err));    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

constexpr size_t NumEdges = 4;

// Graph: A -> C, B -> C, C -> D, A -> D.
static hipGraphNode_t From[NumEdges];
static hipGraphNode_t To[NumEdges];

static bool isExpectedEdge(hipGraphNode_t F, hipGraphNode_t T) {
  for (size_t I = 0; I < NumEdges; I++)
    if (From[I] == F && To[I] == T)
      return true;
  return false;
}

int main() {
  hipGraph_t Graph;
  CHECK(hipGraphCreate(&Graph, 0));
  hipGraphNode_t A, B, C, D;
  CHECK(hipGraphAddEmptyNode(&A, Graph, nullptr, 0));
  CHECK(hipGraphAddEmptyNode(&B, Graph, nullptr, 0));
  CHECK(hipGraphAddEmptyNode(&C, Graph, nullptr, 0));
  CHECK(hipGraphAddEmptyNode(&D, Graph, nullptr, 0));
  From[0] = A;
  To[0] = C;
  From[1] = B;
  To[1] = C;
  From[2] = C;
  To[2] = D;
  From[3] = A;
  To[3] = D;
  CHECK(hipGraphAddDependencies(Graph, From, To, NumEdges));

  // Size query: both arrays null.
  size_t Count = 0;
  CHECK(hipGraphGetEdges(Graph, nullptr, nullptr, &Count));
  if (Count != NumEdges) {
    printf("FAIL: size query returned %zu edges, expected %zu\n", Count,
           NumEdges);
    return 1;
  }

  // Arrays smaller than the edge count: only *numEdges entries may be written.
  // Exactly sized heap arrays so an overflow corrupts the heap and glibc aborts
  // on delete[] (MALLOC_CHECK_=3 in the ctest environment makes that certain).
  {
    size_t Cap = NumEdges - 1;
    hipGraphNode_t *F = new hipGraphNode_t[Cap]();
    hipGraphNode_t *T = new hipGraphNode_t[Cap]();
    CHECK(hipGraphGetEdges(Graph, F, T, &Cap));
    if (Cap != NumEdges - 1) {
      printf("FAIL: numEdges changed to %zu on a short array, expected %zu\n",
             Cap, NumEdges - 1);
      return 1;
    }
    for (size_t I = 0; I < Cap; I++)
      if (!isExpectedEdge(F[I], T[I])) {
        printf("FAIL: short array entry %zu (%p -> %p) is not an edge\n", I,
               (void *)F[I], (void *)T[I]);
        return 1;
      }
    delete[] T;
    delete[] F;
  }

  // Arrays larger than the edge count: surplus entries become null and the
  // actual count is written back.
  {
    size_t Cap = NumEdges + 2;
    hipGraphNode_t Sentinel = reinterpret_cast<hipGraphNode_t>(0x1);
    hipGraphNode_t F[NumEdges + 2], T[NumEdges + 2];
    for (size_t I = 0; I < Cap; I++)
      F[I] = T[I] = Sentinel;
    CHECK(hipGraphGetEdges(Graph, F, T, &Cap));
    if (Cap != NumEdges) {
      printf("FAIL: numEdges left at %zu on a large array, expected %zu\n", Cap,
             NumEdges);
      return 1;
    }
    for (size_t I = 0; I < NumEdges; I++)
      if (!isExpectedEdge(F[I], T[I])) {
        printf("FAIL: large array entry %zu (%p -> %p) is not an edge\n", I,
               (void *)F[I], (void *)T[I]);
        return 1;
      }
    for (size_t I = 0; I < NumEdges; I++) {
      bool Found = false;
      for (size_t J = 0; J < NumEdges; J++)
        Found |= (F[J] == From[I] && T[J] == To[I]);
      if (!Found) {
        printf("FAIL: edge %zu (%p -> %p) missing from the result\n", I,
               (void *)From[I], (void *)To[I]);
        return 1;
      }
    }
    for (size_t I = NumEdges; I < NumEdges + 2; I++)
      if (F[I] != nullptr || T[I] != nullptr) {
        printf("FAIL: surplus entry %zu not null (%p -> %p)\n", I, (void *)F[I],
               (void *)T[I]);
        return 1;
      }
  }

  // Only one of the arrays null is invalid.
  hipGraphNode_t Scratch[NumEdges] = {};
  size_t N = NumEdges;
  if (hipGraphGetEdges(Graph, nullptr, Scratch, &N) != hipErrorInvalidValue ||
      hipGraphGetEdges(Graph, Scratch, nullptr, &N) != hipErrorInvalidValue) {
    printf("FAIL: null from or null to alone must be hipErrorInvalidValue\n");
    return 1;
  }

  CHECK(hipGraphDestroy(Graph));
  printf("PASSED\n");
  return 0;
}
