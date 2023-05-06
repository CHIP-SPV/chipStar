#include <hip/hip_runtime.h>
static __device__ int s = 3;
__device__ int f(int a) { return a + s + 100; }
