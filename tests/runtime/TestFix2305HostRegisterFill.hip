// https://github.com/CHIP-SPV/chipStar/issues/2305
// Fills on a hipHostRegister pointer before hipHostGetDevicePointer must work.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

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

#define REJECT(label, call)                                                    \
  do {                                                                         \
    hipError_t e = (call);                                                     \
    bool ok = e == hipErrorInvalidValue;                                       \
    printf("%-20s %s (%s)\n", label, ok ? "PASSED" : "FAILED",                 \
           hipGetErrorString(e));                                              \
    passed &= ok;                                                              \
  } while (0)

static void slow_overwrite(hipStream_t, hipError_t, void *p) {
  usleep(200000);
  memset(p, 0x88, N);
}

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
  STEP("hipMemset3D slices", hipMemset3D(make_hipPitchedPtr(host, 128, 128, 4), 0x67, make_hipExtent(128, 2, 2)), host[0] == 0x67 && host[255] == 0x67 && host[256] == 0 && host[512] == 0x67 && host[767] == 0x67);
  hipStream_t s;
  e = hipStreamCreate(&s);
  if (e != hipSuccess) {
    printf("hipStreamCreate FAILED (%s)\n", hipGetErrorString(e));
    return 1;
  }

  // A callback that overwrites the host buffer with 0x88 after a delay must
  // complete before a later fill, on the same stream or from the default stream.
  bool same = hipStreamAddCallback(s, slow_overwrite, host, 0) == hipSuccess;
  e = hipMemsetAsync(host, 0x99, N, s);
  same &= e == hipSuccess && hipStreamSynchronize(s) == hipSuccess;
  for (int i = 0; i < N; i++)
    same &= host[i] == 0x99;
  printf("%-20s %s\n", "fill same stream", same ? "PASSED" : "FAILED");
  passed &= same;

  bool cross = hipStreamAddCallback(s, slow_overwrite, host, 0) == hipSuccess;
  e = hipMemset(host, 0x9A, N);
  cross &= e == hipSuccess && hipStreamSynchronize(s) == hipSuccess;
  for (int i = 0; i < N; i++)
    cross &= host[i] == 0x9A;
  printf("%-20s %s\n", "fill cross stream", cross ? "PASSED" : "FAILED");
  passed &= cross;

  REJECT("memset N+1", hipMemset(host, 0, N + 1));
  REJECT("memset -1", hipMemset(host, 0, (size_t)-1));
  REJECT("memset3D range", hipMemset3D(make_hipPitchedPtr(host, 256, 256, 1), 0, make_hipExtent(1, 1, 66)));
  REJECT("memset2D range", hipMemset2D(host + 16000, 1024, 0, 64, 2));
  REJECT("memset3D 2^56", hipMemset3D(make_hipPitchedPtr(host, 128, 1, size_t(1) << 56), 0, make_hipExtent(1, 1, 3)));

  memset(host, 0, N);
  e = hipMemsetAsync(host, 0x77, N, s);
  bool asyncOk = e == hipSuccess;
  asyncOk &= hipStreamSynchronize(s) == hipSuccess;
  asyncOk &= host[0] == 0x77 && host[N - 1] == 0x77;
  printf("%-20s %s\n", "hipMemsetAsync", asyncOk ? "PASSED" : "FAILED");
  passed &= asyncOk;
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
