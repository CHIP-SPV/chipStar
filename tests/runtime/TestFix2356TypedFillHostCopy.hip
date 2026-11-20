// Reproduces https://github.com/CHIP-SPV/chipStar/issues/2356: a typed or
// async fill through the device pointer of a hipHostRegister buffer reaches
// only the device range; the stale host copy is then copied back over it
// before the next kernel. Each case fills a few bytes at an interior offset
// and checks what a kernel reads and what the host copy holds.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(Expr)                                                            \
  do {                                                                         \
    hipError_t E = (Expr);                                                     \
    if (E != hipSuccess) {                                                     \
      printf("FAIL: %s -> %s at line %d\n", #Expr, hipGetErrorString(E),       \
             __LINE__);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

constexpr size_t BufSize = 4096;
constexpr size_t Off = 64;
constexpr size_t Len = 16;
constexpr unsigned char Old = 0xAA;

__global__ void copyBytes(const unsigned char *In, unsigned char *Out) {
  for (int I = 0; I < Len; I++)
    Out[I] = In[I];
}

static int Failures = 0;

// Fill is the Len bytes the call should have written at Off.
static int check(const char *Name, unsigned char *Host, unsigned char *Dev,
                 unsigned char *Out, const unsigned char *Fill) {
  CHECK(hipDeviceSynchronize());
  copyBytes<<<1, 1>>>(Dev + Off, Out);
  CHECK(hipDeviceSynchronize());
  unsigned char Got[Len];
  CHECK(hipMemcpy(Got, Out, Len, hipMemcpyDeviceToHost));
  bool KernelOk = !memcmp(Got, Fill, Len);
  bool HostOk = !memcmp(Host + Off, Fill, Len) && Host[Off - 1] == Old &&
                Host[Off + Len] == Old;
  printf("%-22s kernel %s, host copy %s\n", Name, KernelOk ? "ok" : "WRONG",
         HostOk ? "ok" : "WRONG");
  if (!KernelOk || !HostOk) {
    printf("FAIL case: %s\n", Name);
    Failures++;
  }
  return 0;
}

static int runCase(int Case) {
  const char *Names[] = {"hipMemsetD16",      "hipMemsetD32",
                         "hipMemsetAsync",    "hipMemsetD8Async",
                         "hipMemsetD16Async", "hipMemsetD32Async",
                         "graph memset node"};
  void *Raw;
  if (posix_memalign(&Raw, 4096, BufSize))
    return 1;
  unsigned char *Host = (unsigned char *)Raw;
  memset(Host, Old, BufSize);
  CHECK(hipHostRegister(Host, BufSize, 0));
  unsigned char *Dev;
  CHECK(hipHostGetDevicePointer((void **)&Dev, Host, 0));
  unsigned char *Out;
  CHECK(hipMalloc(&Out, Len));
  hipStream_t S;
  CHECK(hipStreamCreate(&S));

  unsigned char Fill[Len];
  hipDeviceptr_t D = (hipDeviceptr_t)(Dev + Off);
  switch (Case) {
  case 0: {
    unsigned short V = 0x1234;
    for (size_t I = 0; I < Len / 2; I++)
      memcpy(Fill + 2 * I, &V, 2);
    CHECK(hipMemsetD16(D, V, Len / 2));
    break;
  }
  case 1: {
    unsigned V = 0x12345678;
    for (size_t I = 0; I < Len / 4; I++)
      memcpy(Fill + 4 * I, &V, 4);
    CHECK(hipMemsetD32(D, V, Len / 4));
    break;
  }
  case 2:
    memset(Fill, 0x5A, Len);
    CHECK(hipMemsetAsync(Dev + Off, 0x5A, Len, S));
    break;
  case 3:
    memset(Fill, 0x5A, Len);
    CHECK(hipMemsetD8Async(D, 0x5A, Len, S));
    break;
  case 4: {
    unsigned short V = 0x1234;
    for (size_t I = 0; I < Len / 2; I++)
      memcpy(Fill + 2 * I, &V, 2);
    CHECK(hipMemsetD16Async(D, V, Len / 2, S));
    break;
  }
  case 5: {
    unsigned V = 0x12345678;
    for (size_t I = 0; I < Len / 4; I++)
      memcpy(Fill + 4 * I, &V, 4);
    CHECK(hipMemsetD32Async(D, V, Len / 4, S));
    break;
  }
  case 6: {
    memset(Fill, 0x5A, Len);
    hipMemsetParams P = {};
    P.dst = Dev + Off;
    P.elementSize = 1;
    P.height = 1;
    P.pitch = 0;
    P.value = 0x5A;
    P.width = Len;
    hipGraph_t G;
    CHECK(hipGraphCreate(&G, 0));
    hipGraphNode_t N;
    CHECK(hipGraphAddMemsetNode(&N, G, nullptr, 0, &P));
    hipGraphExec_t Ex;
    CHECK(hipGraphInstantiate(&Ex, G, nullptr, nullptr, 0));
    CHECK(hipGraphLaunch(Ex, S));
    break;
  }
  }
  CHECK(hipStreamSynchronize(S));
  if (check(Names[Case], Host, Dev, Out, Fill))
    return 1;

  CHECK(hipStreamDestroy(S));
  CHECK(hipFree(Out));
  CHECK(hipHostUnregister(Host));
  free(Host);
  return 0;
}

int main() {
  int Dev;
  hipDeviceProp_t Prop;
  CHECK(hipGetDevice(&Dev));
  CHECK(hipGetDeviceProperties(&Prop, Dev));
  if (!Prop.canMapHostMemory) {
    printf("HIP_SKIP_THIS_TEST: requires canMapHostMemory\n");
    return CHIP_SKIP_TEST;
  }
  for (int Case = 0; Case < 7; Case++)
    if (runCase(Case))
      return 1;
  printf(Failures ? "FAIL\n" : "PASS\n");
  return Failures ? 1 : 0;
}
