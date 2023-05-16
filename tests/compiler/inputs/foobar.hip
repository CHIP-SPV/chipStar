#include <hip/hip_runtime.h>
__device__ int foo = 100;
__device__ int bar() { return foo + 23; }
__global__ void readFoobar(int *Out) {
  *Out = bar();
  foo = 111;
}
