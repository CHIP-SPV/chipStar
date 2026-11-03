// https://github.com/CHIP-SPV/chipStar/issues/2305
// Fills on a hipHostRegister pointer before hipHostGetDevicePointer must work.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define N 16512

#define STEP(label, call, expect)                                              \
  do {                                                                         \
    memset(host, 0, N);                                                        \
    hipError_t e = (call);                                                     \
    bool ok = e == hipSuccess && (expect);                                     \
    printf("%-20s %s (%s)\n", label, ok ? "PASSED" : "FAILED",                 \
           hipGetErrorString(e));                                              \
    passed &= ok;                                                              \
  } while (0)

__global__ void read_val(unsigned char *out, unsigned char *in, int i) {
  *out = in[i];
}

int main() {
  unsigned char *host = (unsigned char *)malloc(N);
  setvbuf(stdout, NULL, _IONBF, 0);
  bool passed = true;
  hipError_t e = hipHostRegister(host, N, 0);
  printf("hipHostRegister: %s\n", hipGetErrorString(e));
  if (e != hipSuccess)
    return 1;

  STEP("hipMemset", hipMemset(host, 0x11, N), host[0] == 0x11 && host[N - 1] == 0x11);
  STEP("hipMemsetD8", hipMemsetD8((hipDeviceptr_t)host, 0x22, N), host[0] == 0x22 && host[N - 1] == 0x22);
  STEP("hipMemsetD16", hipMemsetD16((hipDeviceptr_t)host, 0x3333, N / 2), host[0] == 0x33 && host[N - 1] == 0x33);
  STEP("hipMemsetD32", hipMemsetD32((hipDeviceptr_t)host, 0x44444444, N / 4), host[0] == 0x44 && host[N - 1] == 0x44);
  STEP("hipMemset2D", hipMemset2D(host, 64, 0x55, 64, 2), host[0] == 0x55 && host[127] == 0x55 && host[128] == 0);
  // Slice stride is pitch * ysize = 128; chipStar rejects depth > pitch.
  STEP("hipMemset3D", hipMemset3D(make_hipPitchedPtr(host, 128, 128, 1), 0x66, make_hipExtent(1, 1, 100)), host[0] == 0x66 && host[99 * 128] == 0x66 && host[1] == 0);
  hipStream_t s;
  hipStreamCreate(&s);
  STEP("hipMemsetAsync", (hipMemsetAsync(host, 0x77, N, s), hipStreamSynchronize(s)), host[0] == 0x77 && host[N - 1] == 0x77);
  hipStreamDestroy(s);

  // The last fill (0x77) must reach the device.
  unsigned char *dev, *res, got = 0;
  e = hipHostGetDevicePointer((void **)&dev, host, 0);
  if (e == hipSuccess && hipMalloc(&res, 1) == hipSuccess) {
    read_val<<<1, 1>>>(res, dev, N - 1);
    hipMemcpy(&got, res, 1, hipMemcpyDeviceToHost);
    hipFree(res);
  }
  bool ok = got == 0x77;
  printf("%-20s %s (got 0x%x)\n", "kernel sees fill", ok ? "PASSED" : "FAILED", got);
  passed &= ok;

  hipHostUnregister(host);
  free(host);
  printf("Overall: %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
