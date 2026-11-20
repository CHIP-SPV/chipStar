// https://github.com/CHIP-SPV/chipStar/issues/2325
// Fills through the host pointer of a hipHostRegister'd buffer, after
// hipHostGetDevicePointer, must reach both the host and device sides.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const size_t N = 4096, Off = 128, Len = 64, Pitch = 256;
static bool AllPassed = true;
static char *Host;
static char *Dev;

__global__ void read_val(int *out, const char *in) { *out = *in; }

static void fill(std::vector<char> &Exp, size_t At, size_t Count,
                 const char *Pat, size_t PatSize) {
  for (size_t i = 0; i < Count; i++)
    Exp[At + i] = Pat[i % PatSize];
}

static void reset() {
  memset(Host, 0, N);
  hipMemset(Dev, 0, N);
  hipDeviceSynchronize();
}

// Compares both sides of the registration against Exp.
static void check(const char *Label, hipError_t Err, hipError_t Want,
                  const std::vector<char> &Exp) {
  std::vector<char> DevCopy(N);
  hipError_t CopyErr = hipMemcpy(DevCopy.data(), Dev, N, hipMemcpyDeviceToHost);
  bool HostOk = !memcmp(Host, Exp.data(), N);
  bool DevOk = CopyErr == hipSuccess && DevCopy == Exp;
  bool Ok = Err == Want && HostOk && DevOk;
  printf("%-28s %s (err=%s host=%s device=%s)\n", Label,
         Ok ? "PASSED" : "FAILED", hipGetErrorString(Err),
         HostOk ? "ok" : "BAD", DevOk ? "ok" : "BAD");
  AllPassed &= Ok;
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int DevId;
  hipDeviceProp_t Prop;
  hipGetDevice(&DevId);
  hipGetDeviceProperties(&Prop, DevId);
  if (!Prop.canMapHostMemory) {
    printf("HIP_SKIP_THIS_TEST: requires canMapHostMemory\n");
    return CHIP_SKIP_TEST;
  }

  // Spare bytes after the registered range so an overrun cannot corrupt the heap.
  Host = (char *)calloc(N + 64, 1);
  if (hipHostRegister(Host, N, 0) != hipSuccess ||
      hipHostGetDevicePointer((void **)&Dev, Host, 0) != hipSuccess) {
    printf("FAILED: register or get device pointer\n");
    return 1;
  }

  const char V = 0x5A;
  std::vector<char> Exp;
  hipError_t E;

  reset();
  Exp.assign(N, 0);
  fill(Exp, Off, Len, &V, 1);
  E = hipMemset(Host + Off, V, Len);
  check("hipMemset(host+off)", E, hipSuccess, Exp);

  reset();
  E = hipMemset(Dev + Off, V, Len);
  check("hipMemset(dev+off)", E, hipSuccess, Exp);

  reset();
  E = hipMemsetD8((hipDeviceptr_t)(Host + Off), V, Len);
  check("hipMemsetD8(host+off)", E, hipSuccess, Exp);

  reset();
  Exp.assign(N, 0);
  unsigned short P16 = 0x1234;
  fill(Exp, Off, Len, (const char *)&P16, 2);
  E = hipMemsetD16((hipDeviceptr_t)(Host + Off), P16, Len / 2);
  check("hipMemsetD16(host+off)", E, hipSuccess, Exp);

  reset();
  Exp.assign(N, 0);
  unsigned P32 = 0xA1B2C3D4;
  fill(Exp, Off, Len, (const char *)&P32, 4);
  E = hipMemsetD32((hipDeviceptr_t)(Host + Off), P32, Len / 4);
  check("hipMemsetD32(host+off)", E, hipSuccess, Exp);

  reset();
  Exp.assign(N, 0);
  fill(Exp, Off, Len, &V, 1);
  E = hipMemsetAsync(Host + Off, V, Len, 0);
  hipError_t S = hipStreamSynchronize(0);
  check("hipMemsetAsync(host+off)", E != hipSuccess ? E : S, hipSuccess, Exp);

  reset();
  Exp.assign(N, 0);
  fill(Exp, Off, Len, &V, 1);
  fill(Exp, Off + Pitch, Len, &V, 1);
  E = hipMemset2D(Host + Off, Pitch, V, Len, 2);
  check("hipMemset2D(host+off)", E, hipSuccess, Exp);

  reset();
  Exp.assign(N, 0);
  for (size_t Z = 0; Z < 2; Z++)
    for (size_t Y = 0; Y < 2; Y++)
      fill(Exp, Off + Z * Pitch * 2 + Y * Pitch, Len, &V, 1);
  hipPitchedPtr PP = make_hipPitchedPtr(Host + Off, Pitch, Pitch, 2);
  E = hipMemset3D(PP, V, make_hipExtent(Len, 2, 2));
  check("hipMemset3D(host+off)", E, hipSuccess, Exp);

  reset();
  Exp.assign(N, 0);
  E = hipMemset(Host + N - 8, V, 16);
  check("hipMemset past the end", E, hipErrorInvalidValue, Exp);

  reset();
  Exp.assign(N, 0);
  fill(Exp, Off, Len, &V, 1);
  int *Res;
  hipMalloc(&Res, sizeof(int));
  E = hipMemset(Host + Off, V, Len);
  read_val<<<1, 1>>>(Res, Dev + Off);
  hipDeviceSynchronize();
  int Got = 0;
  hipMemcpy(&Got, Res, sizeof(int), hipMemcpyDeviceToHost);
  bool KOk = E == hipSuccess && Got == V;
  printf("%-28s %s (err=%s got=%d)\n", "kernel reads filled bytes",
         KOk ? "PASSED" : "FAILED", hipGetErrorString(E), Got);
  AllPassed &= KOk;
  hipFree(Res);

  hipHostUnregister(Host);
  free(Host);
  printf("Overall: %s\n", AllPassed ? "PASSED" : "FAILED");
  return AllPassed ? 0 : 1;
}
