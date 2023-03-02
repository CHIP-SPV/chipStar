#include <hip/hip_runtime.h>
#include <iostream>

extern __global__ void readFoobar(int *Out); // Defined in inputs/foobar.hip.

__device__ int foo = 300;
__device__ int bar() { return foo + 21; }
__global__ void readFoobar2(int *Out) {
  *Out = bar();
  foo = 222;
}

int main() {
  int *OutD, OutH;
  (void)hipMalloc(&OutD, sizeof(int));

  readFoobar2<<<1, 1>>>(OutD);
  (void)hipMemcpy(&OutH, OutD, sizeof(int), hipMemcpyDeviceToHost);
  if (OutH != 321)
    return 1;

  readFoobar<<<1, 1>>>(OutD);
  (void)hipMemcpy(&OutH, OutD, sizeof(int), hipMemcpyDeviceToHost);
  if (OutH != 123)
    return 2;

  (void)hipMemcpyFromSymbol(&OutH, foo, sizeof(int), 0, hipMemcpyDeviceToHost);
  printf("OutH=%d\n", OutH);
  if (OutH != 222)
    return 3;

  return 0;
}
