// Functional test: hipHostRegister + hipGetDevicePointer + kernel write-back.
// Registers malloc'd host memory, obtains the device pointer, runs a kernel
// that writes 1 to it, then verifies the host side sees the change.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(cmd)                                                             \
  do {                                                                         \
    hipError_t e = (cmd);                                                      \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error %s at %s:%d\n",                              \
              hipGetErrorString(e), __FILE__, __LINE__);                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void set_one(int *ptr) { *ptr = 1; }

int main() {
  hipDeviceProp_t prop;
  int dev;
  CHECK(hipGetDevice(&dev));
  CHECK(hipGetDeviceProperties(&prop, dev));
  if (!prop.canMapHostMemory) {
    printf("HIP_SKIP_THIS_TEST: requires canMapHostMemory\n");
    return CHIP_SKIP_TEST;
  }

  int *host = (int *)malloc(sizeof(int));
  *host = 0;

  CHECK(hipHostRegister(host, sizeof(int), hipHostRegisterMapped));

  int *device;
  CHECK(hipHostGetDevicePointer((void **)&device, host, 0));

  set_one<<<dim3(1), dim3(1)>>>(device);
  CHECK(hipDeviceSynchronize());

  bool passed = (*host == 1);
  printf("*host = %d — %s\n", *host, passed ? "PASSED" : "FAILED");

  CHECK(hipHostUnregister(host));
  free(host);

  return passed ? 0 : 1;
}
