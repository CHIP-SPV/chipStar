// Reproduces CHIP-SPV/chipStar#1736: __float2half_rd rounds to nearest, not down.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

__global__ void conv(float X, unsigned short *Out) {
  *Out = __half_as_ushort(__float2half_rd(-X));
}

int main() {
  // -X lies between halves 0xbc01 and 0xbc02, nearer 0xbc01.
  float X = 1.0009765625f + 0.0002f;
  unsigned short H = 0;
  unsigned short *DH;
  (void)hipMalloc(&DH, sizeof(H));
  conv<<<1, 1>>>(X, DH);
  (void)hipMemcpy(&H, DH, sizeof(H), hipMemcpyDeviceToHost);
  bool Ok = H == 0xbc02;
  printf("%s: got %#x, expected 0xbc02\n", Ok ? "PASSED" : "FAILED", H);
  return Ok ? 0 : 1;
}
