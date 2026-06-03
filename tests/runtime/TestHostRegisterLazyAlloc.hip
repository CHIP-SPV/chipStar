// Tests for lazy device allocation in hipHostRegister:
//   1. hipHostGetDevicePointer called twice returns the same pointer.
//   2. hipHostUnregister without hipHostGetDevicePointer does not crash or leak.
//   3. hipMemset via the device pointer after hipHostGetDevicePointer is
//      visible to a subsequent kernel.
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

#define REPORT(label, cond)                                                    \
  do {                                                                         \
    printf("%-55s %s\n", label, (cond) ? "PASSED" : "FAILED");                \
    passed &= (cond);                                                          \
  } while (0)

__global__ void read_val(int *out, int *in) { *out = *in; }

int main() {
  hipDeviceProp_t prop;
  int dev;
  CHECK(hipGetDevice(&dev));
  CHECK(hipGetDeviceProperties(&prop, dev));
  if (!prop.canMapHostMemory) {
    printf("HIP_SKIP_THIS_TEST: requires canMapHostMemory\n");
    return CHIP_SKIP_TEST;
  }

  bool passed = true;

  // --- 1. hipHostGetDevicePointer called twice returns the same pointer ------
  {
    int *host = (int *)malloc(sizeof(int));
    CHECK(hipHostRegister(host, sizeof(int), hipHostRegisterMapped));

    int *devA, *devB;
    CHECK(hipHostGetDevicePointer((void **)&devA, host, 0));
    CHECK(hipHostGetDevicePointer((void **)&devB, host, 0));

    REPORT("GetDevicePointer x2 returns same pointer", devA == devB);

    CHECK(hipHostUnregister(host));
    free(host);
  }

  // --- 2. Unregister without GetDevicePointer (no device allocation) ---------
  {
    int *host = (int *)malloc(sizeof(int));
    CHECK(hipHostRegister(host, sizeof(int), hipHostRegisterMapped));
    CHECK(hipHostUnregister(host));
    free(host);
    printf("%-55s %s\n", "Unregister without GetDevicePointer", "PASSED");
  }

  // --- 3. hipMemset after GetDevicePointer is visible to a kernel ------------
  {
    int *host = (int *)malloc(sizeof(int));
    *host = 99;
    CHECK(hipHostRegister(host, sizeof(int), hipHostRegisterMapped));

    int *device;
    CHECK(hipHostGetDevicePointer((void **)&device, host, 0));

    CHECK(hipMemset(device, 0, sizeof(int)));

    int *result;
    CHECK(hipMalloc(&result, sizeof(int)));
    read_val<<<dim3(1), dim3(1)>>>(result, device);
    CHECK(hipDeviceSynchronize());

    int got;
    CHECK(hipMemcpy(&got, result, sizeof(int), hipMemcpyDeviceToHost));
    REPORT("hipMemset after GetDevicePointer visible to kernel", got == 0);

    CHECK(hipFree(result));
    CHECK(hipHostUnregister(host));
    free(host);
  }

  printf("\nOverall: %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
