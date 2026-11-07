// Reproduces #2317: a graph memset node fills Height * Width bytes and
// ignores elementSize, although Width counts elements.
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

constexpr size_t Bytes = 256;
static unsigned char *Dev;

// Compare the whole allocation: Pattern repeated over Filled bytes, 0 after.
static int verify(const char *Name, unsigned Value, size_t ElemSize,
                  size_t Filled) {
  unsigned char Host[Bytes];
  CHECK(hipMemcpy(Host, Dev, Bytes, hipMemcpyDeviceToHost));
  for (size_t I = 0; I < Bytes; I++) {
    unsigned char Want = 0;
    if (I < Filled)
      Want = ((unsigned char *)&Value)[I % ElemSize];
    if (Host[I] != Want) {
      printf("FAIL: %s: byte %zu is 0x%02x, expected 0x%02x (%zu bytes should "
             "be filled)\n",
             Name, I, Host[I], Want, Filled);
      return 1;
    }
  }
  return 0;
}

static int runGraph(hipGraph_t G, hipStream_t S) {
  hipGraphExec_t E;
  CHECK(hipGraphInstantiate(&E, G, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(E, S));
  CHECK(hipStreamSynchronize(S));
  CHECK(hipGraphExecDestroy(E));
  return 0;
}

int main() {
  CHECK(hipMalloc((void **)&Dev, Bytes));
  hipStream_t S;
  CHECK(hipStreamCreate(&S));

  // Part A: explicit memset nodes.
  struct {
    size_t ElemSize, Width, Height;
    unsigned Value;
  } Cases[] = {{2, 8, 1, 0x1234},
               {4, 8, 1, 0x12345678},
               {4, 4, 2, 0x12345678},
               {1, 8, 1, 0x5a}};
  for (auto &C : Cases) {
    CHECK(hipMemset(Dev, 0, Bytes));
    hipGraph_t G;
    CHECK(hipGraphCreate(&G, 0));
    hipMemsetParams P = {};
    P.dst = Dev;
    P.elementSize = C.ElemSize;
    P.width = C.Width;
    P.height = C.Height;
    P.pitch = C.Width * C.ElemSize;
    P.value = C.Value;
    hipGraphNode_t N;
    CHECK(hipGraphAddMemsetNode(&N, G, nullptr, 0, &P));
    if (runGraph(G, S))
      return 1;
    char Name[64];
    snprintf(Name, sizeof Name, "node elem=%zu w=%zu h=%zu", C.ElemSize,
             C.Width, C.Height);
    if (verify(Name, C.Value, C.ElemSize, C.ElemSize * C.Width * C.Height))
      return 1;
    CHECK(hipGraphDestroy(G));
  }

  // Part B: stream capture of hipMemsetD16Async / hipMemsetD32Async.
  for (int Is32 = 0; Is32 < 2; Is32++) {
    CHECK(hipMemset(Dev, 0, Bytes));
    hipGraph_t G;
    CHECK(hipStreamBeginCapture(S, hipStreamCaptureModeGlobal));
    if (Is32)
      CHECK(hipMemsetD32Async((hipDeviceptr_t)Dev, 0x12345678, 8, S));
    else
      CHECK(hipMemsetD16Async((hipDeviceptr_t)Dev, 0x1234, 8, S));
    CHECK(hipStreamEndCapture(S, &G));
    if (runGraph(G, S))
      return 1;
    if (Is32 ? verify("capture D32", 0x12345678, 4, 32)
             : verify("capture D16", 0x1234, 2, 16))
      return 1;
    CHECK(hipGraphDestroy(G));
  }

  printf("PASS\n");
  return 0;
}
