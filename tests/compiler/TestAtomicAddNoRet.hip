#include <hip/hip_runtime.h>
__global__ void k(float *d) { atomicAddNoRet(d, 1.0f); }
