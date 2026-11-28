// Reproduces #2302: hipMemset2D rejected Width > Pitch although pitch is unused when Height is 1.
#include <hip/hip_runtime.h>
#include <cstdio>

static int Bad;
#define CHECK(Call, Want)                                                      \
  do {                                                                         \
    hipError_t E = (Call);                                                     \
    if (E != (Want)) {                                                         \
      fprintf(stderr, "FAIL: line %d: %s returned %d, expected %d\n", __LINE__, \
              #Call, (int)E, (int)(Want));                                     \
      fflush(stderr);                                                          \
      Bad = 1;                                                                 \
    }                                                                          \
  } while (0)

int main() {
  unsigned char *P, H[64];
  if (hipMalloc(&P, 64) != hipSuccess)
    return 1;
  CHECK(hipMemset(P, 0, 64), hipSuccess);
  CHECK(hipMemset2D(P, 1, 0xAB, 7, 1), hipSuccess);
  CHECK(hipMemcpy(H, P, 64, hipMemcpyDeviceToHost), hipSuccess);
  for (int I = 0; I < 64; I++)
    if (H[I] != (I < 7 ? 0xAB : 0)) {
      fprintf(stderr, "FAIL: byte %d is 0x%02x\n", I, H[I]);
      Bad = 1;
      break;
    }
  CHECK(hipMemset2D(P, 1, 0xAB, 65, 1), hipErrorInvalidValue);
  CHECK(hipMemset2D(P, 1, 0xAB, 7, 2), hipErrorInvalidValue);
  CHECK(hipMemset2D(P, 0, 0xCD, 7, 1), hipSuccess);
  CHECK(hipMemset2D(P, 0, 0, 0, 1), hipSuccess);
  if (Bad)
    return 1;
  printf("PASS\n");
  return 0;
}
