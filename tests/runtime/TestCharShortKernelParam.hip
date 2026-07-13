#include <hip/hip_runtime.h>
#include <cstdio>

// Regression test for issue #1261: char (i8) and short (i16) kernel arguments
// are valid SPIR-V OpTypeInt widths (8, 16, 32, 64) and must NOT be promoted
// to i32. This checks that narrow integer parameters are passed and used
// correctly by the runtime.

__global__ void charKernel(char c, int *out) { *out = (int)c; }

__global__ void shortKernel(short s, int *out) { *out = (int)s; }

__global__ void unsignedNarrowKernel(unsigned char uc, unsigned short us,
                                     int *out) {
  *out = (int)uc + (int)us;
}

__global__ void mixedKernel(char a, short b, int c, int *out) {
  *out = (int)a + (int)b + c;
}

int main() {
  int *d_out;
  int h_out;
  hipMalloc(&d_out, sizeof(int));

  charKernel<<<1, 1>>>((char)-7, d_out);
  hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost);
  if (h_out != -7) {
    printf("FAIL: charKernel(-7) got %d\n", h_out);
    return 1;
  }

  shortKernel<<<1, 1>>>((short)-1234, d_out);
  hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost);
  if (h_out != -1234) {
    printf("FAIL: shortKernel(-1234) got %d\n", h_out);
    return 1;
  }

  unsignedNarrowKernel<<<1, 1>>>((unsigned char)200, (unsigned short)60000,
                                 d_out);
  hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost);
  if (h_out != 200 + 60000) {
    printf("FAIL: unsignedNarrowKernel got %d\n", h_out);
    return 1;
  }

  mixedKernel<<<1, 1>>>((char)5, (short)500, 50000, d_out);
  hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost);
  if (h_out != 5 + 500 + 50000) {
    printf("FAIL: mixedKernel got %d\n", h_out);
    return 1;
  }

  printf("PASS\n");
  hipFree(d_out);
  return 0;
}
