#ifndef TEST_STATIC_LIB_RDC_K_H
#define TEST_STATIC_LIB_RDC_K_H

#include "hip/hip_runtime.h"
#include <cassert>

__device__ int device_square(int x);
void test(float *d_x,float *d_y, float *x, float *y, int N );
void test2(float *d_x,float *d_y, float *x, float *y, int N );

#endif // TEST_STATIC_LIB_RDC_K_H 