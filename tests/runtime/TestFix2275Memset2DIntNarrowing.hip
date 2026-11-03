// Reproduces #2275: hipMemset2D bounds check narrowed to int and rejected fills that fit.
#include <hip/hip_runtime_api.h>
#include <cstdio>

static int Failed = 0;

static void check(const char *Name, bool Ok) {
  if (!Ok) {
    printf("FAIL: %s\n", Name);
    Failed++;
  }
}

// Fills Pitch 16, Width 7, Height 2 and checks only those 14 bytes changed.
static void fillAndVerify(const char *Name, void *P) {
  hipError_t Err = hipMemset2D(P, 16, 0xAB, 7, 2);
  check(Name, Err == hipSuccess);
  if (Err != hipSuccess)
    return;
  unsigned char Out[23];
  check(Name, hipMemcpy(Out, P, 23, hipMemcpyDeviceToHost) == hipSuccess);
  for (int I = 0; I < 23; I++) {
    bool Inside = I < 7 || I >= 16;
    if (Out[I] != (Inside ? 0xAB : 0x00)) {
      printf("FAIL: %s byte %d is 0x%02x\n", Name, I, Out[I]);
      Failed++;
    }
  }
}

int main() {
  void *P, *Q, *Big;
  if (hipMalloc(&P, 23) != hipSuccess || hipMalloc(&Q, 22) != hipSuccess)
    return 1;
  hipMemset(P, 0, 23);

  fillAndVerify("a) exact fit, 23 bytes", P);
  check("b) one byte too small, 22 bytes",
        hipMemset2D(Q, 16, 0xAB, 7, 2) == hipErrorInvalidValue);
  check("c) Pitch * (Height - 1) wraps to 0",
        hipMemset2D(P, size_t(1) << 32, 0xAB, 1, (size_t(1) << 32) + 1) ==
            hipErrorInvalidValue);
  hipFree(P);
  hipFree(Q);

  hipError_t Err = hipMalloc(&Big, size_t(1) << 31);
  if (Err != hipSuccess) {
    printf("SKIP: hipMalloc of 2 GiB failed: %s\n", hipGetErrorString(Err));
    return Failed ? 1 : CHIP_SKIP_TEST;
  }
  hipMemset(Big, 0, 23);
  fillAndVerify("d) 2 GiB allocation", Big);
  hipFree(Big);

  if (Failed)
    return 1;
  printf("PASS\n");
  return 0;
}
