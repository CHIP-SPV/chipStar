// Reproduces https://github.com/CHIP-SPV/chipStar/issues/2312: Level0 hipMemset3D truncates a slice pitch above UINT32_MAX.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

#define CHECK(X)                                                               \
  do {                                                                         \
    hipError_t E = (X);                                                        \
    if (E != hipSuccess) {                                                     \
      printf("FAIL: %s returned %d\n", #X, (int)E);                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  hipPitchedPtr P;
  if (hipMalloc3D(&P, make_hipExtent(128, 33554433, 2)) != hipSuccess)
    return CHIP_SKIP_TEST;

  size_t SlicePitch = P.pitch * P.ysize;
  size_t Narrowed = (uint32_t)SlicePitch;
  char *Base = (char *)P.ptr;
  CHECK(hipMemset(Base + Narrowed, 0, 1));
  CHECK(hipMemset(Base + SlicePitch, 0, 1));

  CHECK(hipMemset3D(P, 0xAB, make_hipExtent(127, 1, 2)));
  CHECK(hipDeviceSynchronize());

  unsigned char AtTrue = 0, AtNarrowed = 0xFF;
  CHECK(hipMemcpy(&AtTrue, Base + SlicePitch, 1, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(&AtNarrowed, Base + Narrowed, 1, hipMemcpyDeviceToHost));
  CHECK(hipFree(P.ptr));

  printf("slice pitch %zu narrowed %zu: byte at true offset 0x%02X (want 0xAB), "
         "byte at narrowed offset 0x%02X (want 0x00)\n",
         SlicePitch, Narrowed, AtTrue, AtNarrowed);
  if (AtTrue != 0xAB || AtNarrowed != 0) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}
