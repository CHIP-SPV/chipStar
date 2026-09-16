// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1647: device
// variable names that, after __chip_var_, spell a shadow kernel name.
#include <hip/hip_runtime.h>

__device__ int x = 1, info_x = 2, bind_x = 3, init_x = 4, init_all = 5;

__global__ void k(int *O) { O[0] = x + info_x + bind_x + init_x + init_all; }
