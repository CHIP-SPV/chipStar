// __float2half_rz/_ru/_rd round in their own direction, not to nearest.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %s (line %d)\n", #Call, hipGetErrorName(Err),  \
             __LINE__);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void conv(const float *X, unsigned short *Out) {
  Out[0] = __half_as_ushort(__float2half_rz(X[0]));
  Out[1] = __half_as_ushort(__float2half_ru(X[0]));
  Out[2] = __half_as_ushort(__float2half_rd(X[0]));
  Out[3] = __half_as_ushort(__float2half_rd(-X[0]));
  Out[4] = __half_as_ushort(__float2half_rz(X[1]));
}

int main() {
  // Just above 1 + 2^-10 (0x3c01), well below the next half (0x3c02);
  // and above the largest half, which rounds toward zero to 65504.
  float X[2] = {1.0009765625f + 0.0002f, 70000.0f};
  unsigned short Exp[5] = {0x3c01, 0x3c02, 0x3c01, 0xbc02, 0x7bff}, H[5];
  float *DX;
  unsigned short *DH;
  CHECK(hipMalloc(&DX, sizeof(X)));
  CHECK(hipMalloc(&DH, sizeof(H)));
  CHECK(hipMemcpy(DX, X, sizeof(X), hipMemcpyHostToDevice));
  conv<<<1, 1>>>(DX, DH);
  CHECK(hipGetLastError());
  CHECK(hipMemcpy(H, DH, sizeof(H), hipMemcpyDeviceToHost));
  int Bad = 0;
  for (int I = 0; I < 5; ++I)
    if (H[I] != Exp[I] && ++Bad)
      printf("case %d: %#x, expected %#x\n", I, H[I], Exp[I]);
  printf("%s\n", Bad ? "FAILED" : "PASSED");
  return Bad != 0;
}
