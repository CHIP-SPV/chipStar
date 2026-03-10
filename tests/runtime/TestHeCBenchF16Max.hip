// Minimal reproducer for chipStar issue #1179:
// __byte_perm intrinsic produces wrong results, breaking the HeCBench
// f16max benchmark's half-precision max via bit manipulation.
//
// __byte_perm(a, b, selector):
//   Treats a as bytes 0-3 and b as bytes 4-7 (little-endian).
//   Each nibble of selector (low to high) picks one byte index (0-7)
//   for the corresponding output byte position (low to high).

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__device__ unsigned int byte_perm_wrapper(unsigned int a, unsigned int b,
                                          unsigned int s) {
  return __byte_perm(a, b, s);
}

__global__ void testBytePerm(unsigned int *results, const unsigned int *a,
                             const unsigned int *b, const unsigned int *sel,
                             int n) {
  int i = threadIdx.x;
  if (i < n)
    results[i] = byte_perm_wrapper(a[i], b[i], sel[i]);
}

int main() {
  // Byte layout for a = 0x03020100: byte0=0x00 byte1=0x01 byte2=0x02 byte3=0x03
  // Byte layout for b = 0x07060504: byte4=0x04 byte5=0x05 byte6=0x06 byte7=0x07

  const int N = 8;
  unsigned int h_a[N], h_b[N], h_sel[N], h_expected[N], h_results[N];

  // Test 0: identity from a  (sel=0x3210 -> bytes 0,1,2,3 from a)
  h_a[0] = 0x03020100; h_b[0] = 0x07060504; h_sel[0] = 0x3210;
  h_expected[0] = 0x03020100;

  // Test 1: identity from b  (sel=0x7654 -> bytes 4,5,6,7 from b)
  h_a[1] = 0x03020100; h_b[1] = 0x07060504; h_sel[1] = 0x7654;
  h_expected[1] = 0x07060504;

  // Test 2: byte-reverse a  (sel=0x0123 -> bytes 3,2,1,0)
  h_a[2] = 0x03020100; h_b[2] = 0x07060504; h_sel[2] = 0x0123;
  h_expected[2] = 0x00010203;

  // Test 3: mix bytes from a and b  (sel=0x5410 -> byte0,byte1,byte4,byte5)
  // out_byte0 = byte0 = 0x00
  // out_byte1 = byte1 = 0x01
  // out_byte2 = byte4 = 0x04
  // out_byte3 = byte5 = 0x05
  h_a[3] = 0x03020100; h_b[3] = 0x07060504; h_sel[3] = 0x5410;
  h_expected[3] = 0x05040100;

  // Test 4: all same byte from a  (sel=0x0000 -> byte0 four times)
  // a = 0xAABBCCDD: byte0=0xDD
  h_a[4] = 0xAABBCCDD; h_b[4] = 0x11223344; h_sel[4] = 0x0000;
  h_expected[4] = 0xDDDDDDDD;

  // Test 5: all same byte from b  (sel=0x4444 -> byte4 four times)
  // b = 0x11223344: byte4=0x44
  h_a[5] = 0xAABBCCDD; h_b[5] = 0x11223344; h_sel[5] = 0x4444;
  h_expected[5] = 0x44444444;

  // Test 6: half_max pattern - a > b case (sel=0x3210, select a)
  // This is the selector the HeCBench f16max kernel produces when a >= b.
  // a = two fp16 infinities packed as half2 = 0x7C007C00
  // b = zero = 0x00000000
  // max(a,b) should be a -> __byte_perm returns a unchanged
  h_a[6] = 0x7C007C00; h_b[6] = 0x00000000; h_sel[6] = 0x3210;
  h_expected[6] = 0x7C007C00;

  // Test 7: half_max pattern - a < b case (sel=0x7654, select b)
  // When a < b, the HeCBench kernel computes sel=0x7654.
  // a = zero, b = two fp16 infinities
  // max(a,b) should be b -> __byte_perm returns b unchanged
  h_a[7] = 0x00000000; h_b[7] = 0x7C007C00; h_sel[7] = 0x7654;
  h_expected[7] = 0x7C007C00;

  unsigned int *d_a, *d_b, *d_sel, *d_results;
  hipMalloc(&d_a, N * sizeof(unsigned int));
  hipMalloc(&d_b, N * sizeof(unsigned int));
  hipMalloc(&d_sel, N * sizeof(unsigned int));
  hipMalloc(&d_results, N * sizeof(unsigned int));

  hipMemcpy(d_a, h_a, N * sizeof(unsigned int), hipMemcpyHostToDevice);
  hipMemcpy(d_b, h_b, N * sizeof(unsigned int), hipMemcpyHostToDevice);
  hipMemcpy(d_sel, h_sel, N * sizeof(unsigned int), hipMemcpyHostToDevice);

  testBytePerm<<<1, N>>>(d_results, d_a, d_b, d_sel, N);
  hipDeviceSynchronize();
  hipMemcpy(h_results, d_results, N * sizeof(unsigned int),
            hipMemcpyDeviceToHost);

  bool pass = true;
  for (int i = 0; i < N; i++) {
    if (h_results[i] != h_expected[i]) {
      printf("FAIL: test %d: __byte_perm(0x%08X, 0x%08X, 0x%04X) = 0x%08X, "
             "expected 0x%08X\n",
             i, h_a[i], h_b[i], h_sel[i], h_results[i], h_expected[i]);
      pass = false;
    }
  }

  printf("%s\n", pass ? "PASSED" : "FAILED");

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_sel);
  hipFree(d_results);
  return pass ? 0 : 1;
}
