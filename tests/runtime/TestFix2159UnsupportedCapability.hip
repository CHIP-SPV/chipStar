// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2159: a module
// that declares capability Float64 must be rejected on a device without
// shaderFloat64 (hasDoubles == 0) with hipErrorSharedObjectInitFailed instead
// of being passed to the driver, whose own failure is a different error. A
// kernel without doubles in the same module must still run.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void i(int *Out) { *Out = 7; }
__global__ void k(double *Out, double X) { *Out = X * 3.0; }

int main() {
  hipDeviceProp_t P;
  if (hipGetDeviceProperties(&P, 0) != hipSuccess)
    return 1;
  double *D = nullptr;
  int *I = nullptr, HI = 0;
  if (hipMalloc(&D, sizeof(double)) != hipSuccess ||
      hipMalloc(&I, sizeof(int)) != hipSuccess)
    return 1;
  i<<<1, 1>>>(I);
  if (hipDeviceSynchronize() != hipSuccess ||
      hipMemcpy(&HI, I, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess ||
      HI != 7) {
    printf("FAILED: a kernel without doubles did not run, got %d\n", HI);
    return 1;
  }
  k<<<1, 1>>>(D, 2.0);
  hipError_t Err = hipGetLastError();
  if (Err == hipSuccess)
    Err = hipDeviceSynchronize();
  double H = 0;
  if (!P.arch.hasDoubles) {
    if (Err != hipErrorSharedObjectInitFailed) {
      printf("FAILED: Float64 module on a device without shaderFloat64 gave "
             "error %d, not module rejection\n", (int)Err);
      return 1;
    }
  } else if (Err != hipSuccess ||
             hipMemcpy(&H, D, sizeof(H), hipMemcpyDeviceToHost) != hipSuccess ||
             H != 6.0) {
    printf("FAILED: hasDoubles but result %f, err %d\n", H, (int)Err);
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
