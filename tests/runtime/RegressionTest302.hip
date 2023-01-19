#include "hip/hip_runtime.h"

// CHIP-SPV/chip-spv#302
__global__ static void repro_kernel(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int value_as_ull = *address_as_ull;
  atomicCAS(address_as_ull, value_as_ull,
            __double_as_longlong(val + __longlong_as_double(value_as_ull)));
}

int main() {
  double *mem_d, mem_h = 1.5;
  (void)hipMalloc(&mem_d, sizeof(double));
  (void)hipMemcpy(mem_d, &mem_h, sizeof(double), hipMemcpyHostToDevice);
  repro_kernel<<<1, 1>>>(mem_d, 1.0);
  (void)hipMemcpy(&mem_h, mem_d, sizeof(double), hipMemcpyDeviceToHost);
  (void)hipFree(mem_d);
  return !(mem_h == 2.5f);
}
