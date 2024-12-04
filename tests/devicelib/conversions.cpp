#include <hip/hip_runtime.h>
#include <cassert>

// Define a new kernel for testing __double2hiint
__global__ void testDouble2HiIntKernel(double *input, int *output) {
  *output = __double2hiint(*input);
}

// Test __double2hiint in a kernel
int testDouble2HiInt() {
  double testVal = 3.14159265359;
  int highBits;
  double *d_testVal;
  int *d_highBits;

  // Allocate memory on the device
  (void)hipMalloc(&d_testVal, sizeof(double));
  (void)hipMalloc(&d_highBits, sizeof(int));

  // Copy the test value to the device
  (void)hipMemcpy(d_testVal, &testVal, sizeof(double), hipMemcpyHostToDevice);

  // Launch the kernel
  testDouble2HiIntKernel<<<1, 1>>>(d_testVal, d_highBits);

  // Copy the result back to the host
  (void)hipMemcpy(&highBits, d_highBits, sizeof(int), hipMemcpyDeviceToHost);

  // Free device memory
  (void)hipFree(d_testVal);
  (void)hipFree(d_highBits);

  // The high 32 bits of 3.14159265359 in double precision should be 0x400921fb
  assert(highBits == 0x400921fb);

  return 0;
}

int main() {
  return testDouble2HiInt();
}
