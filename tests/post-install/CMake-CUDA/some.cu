#include <cstdio>
#include <cuda_runtime.h>
__global__ void world() { printf(", World!\n"); }
