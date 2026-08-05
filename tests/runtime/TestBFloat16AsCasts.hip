// __short_as_bfloat16 and __ushort_as_bfloat16 must reinterpret the bits, not
// convert the value (CHIP-SPV/chipStar#1385).
//
// They used to build the result as __hip_bfloat16{a}. chipStar's
// __hip_bfloat16 is not an aggregate, it has a converting constructor from
// float, so that selected the constructor and __ushort_as_bfloat16(0x3F80)
// returned 16256.0 instead of 1.0. Including <hip/hip_bf16.h> at all from a
// plain -I path was a hard error for the same reason, since the conversion is
// narrowing, so this file doubles as the compile check.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdio>

__global__ void asCasts(float *Out, unsigned short *Bits) {
  // 0x3F80 is the bfloat16 encoding of 1.0f, 0xC000 that of -2.0f.
  __hip_bfloat16 A = __ushort_as_bfloat16(0x3F80);
  __hip_bfloat16 B = __short_as_bfloat16((short)0xC000);
  Out[0] = __bfloat162float(A);
  Out[1] = __bfloat162float(B);
  Bits[0] = __bfloat16_as_ushort(A);
  Bits[1] = (unsigned short)__bfloat16_as_short(B);
}

int main() {
  float *DOut = nullptr;
  unsigned short *DBits = nullptr;
  (void)hipMalloc(&DOut, 2 * sizeof(float));
  (void)hipMalloc(&DBits, 2 * sizeof(unsigned short));

  asCasts<<<1, 1>>>(DOut, DBits);
  (void)hipDeviceSynchronize();

  float HOut[2] = {0.0f, 0.0f};
  unsigned short HBits[2] = {0, 0};
  (void)hipMemcpy(HOut, DOut, sizeof(HOut), hipMemcpyDeviceToHost);
  (void)hipMemcpy(HBits, DBits, sizeof(HBits), hipMemcpyDeviceToHost);

  bool Ok = true;
  if (HOut[0] != 1.0f || HOut[1] != -2.0f) {
    printf("FAIL: values %f, %f (want 1.000000, -2.000000)\n", HOut[0],
           HOut[1]);
    Ok = false;
  }
  if (HBits[0] != 0x3F80 || HBits[1] != 0xC000) {
    printf("FAIL: bits 0x%04X, 0x%04X (want 0x3F80, 0xC000)\n", HBits[0],
           HBits[1]);
    Ok = false;
  }

  (void)hipFree(DOut);
  (void)hipFree(DBits);

  if (Ok)
    printf("PASSED\n");
  return Ok ? 0 : 1;
}
