// hipMemset on a mapped hipHostMalloc allocation must keep earlier host stores: https://github.com/CHIP-SPV/chipStar/issues/2365

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

#define CHECK(Expr)                                                            \
  do {                                                                         \
    hipError_t Err = (Expr);                                                   \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %s\n", #Expr, hipGetErrorString(Err));         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void copyOut(const unsigned char *P, unsigned char *Out) {
  Out[0] = P[0];
  Out[1] = P[1024];
  Out[2] = P[1040];
  Out[3] = P[2000];
}

static bool expect(const char *What, const char *Where, size_t Off,
                   unsigned char Got, unsigned char Want) {
  if (Got == Want)
    return true;
  printf("FAIL: %s %s p[%zu] = 0x%02x, expected 0x%02x\n", What, Where, Off,
         Got, Want);
  return false;
}

static int run(const char *What, int Kind) {
  unsigned char *P = nullptr;
  unsigned char *Out = nullptr;
  CHECK(hipHostMalloc((void **)&P, 4096, hipHostMallocMapped));
  CHECK(hipMalloc((void **)&Out, 4));

  memset(P, 0xAA, 4096);
  if (Kind == 0)
    CHECK(hipMemset(P + 1024, 0x55, 16));
  else if (Kind == 1)
    CHECK(hipMemsetD8((hipDeviceptr_t)(P + 1024), 0x55, 16));
  else if (Kind == 2)
    CHECK(hipMemsetD16((hipDeviceptr_t)(P + 1024), 0x5555, 8));
  else
    CHECK(hipMemsetD32((hipDeviceptr_t)(P + 1024), 0x55555555, 4));

  bool Ok = expect(What, "host", 0, P[0], 0xAA);
  Ok &= expect(What, "host", 1024, P[1024], 0x55);
  Ok &= expect(What, "host", 1039, P[1039], 0x55);
  Ok &= expect(What, "host", 1040, P[1040], 0xAA);
  Ok &= expect(What, "host", 4095, P[4095], 0xAA);

  // The allocation must be mapped for the host again after the call.
  P[2000] = 0x11;

  copyOut<<<1, 1>>>(P, Out);
  unsigned char Seen[4] = {0, 0, 0, 0};
  CHECK(hipMemcpy(Seen, Out, 4, hipMemcpyDeviceToHost));
  Ok &= expect(What, "device", 0, Seen[0], 0xAA);
  Ok &= expect(What, "device", 1024, Seen[1], 0x55);
  Ok &= expect(What, "device", 1040, Seen[2], 0xAA);
  Ok &= expect(What, "device", 2000, Seen[3], 0x11);

  Ok &= expect(What, "host after launch", 2000, P[2000], 0x11);
  Ok &= expect(What, "host after launch", 1024, P[1024], 0x55);

  CHECK(hipFree(Out));
  CHECK(hipHostFree(P));
  return Ok ? 0 : 1;
}

int main() {
  hipDeviceProp_t Props;
  CHECK(hipGetDeviceProperties(&Props, 0));
  if (!Props.canMapHostMemory) {
    printf("HIP_SKIP_THIS_TEST: canMapHostMemory == 0\n");
    return CHIP_SKIP_TEST;
  }
  int Rc = run("hipMemset", 0);
  Rc |= run("hipMemsetD8", 1);
  Rc |= run("hipMemsetD16", 2);
  Rc |= run("hipMemsetD32", 3);
  printf("%s\n", Rc ? "FAIL" : "PASS");
  return Rc;
}
