#include <stdio.h>
#include <math.h>
#include <stdlib.h> // Added for malloc
#include "k.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

int main(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;

  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  HIP_ASSERT(hipMalloc(&d_x, N*sizeof(float)));
  HIP_ASSERT(hipMalloc(&d_y, N*sizeof(float)));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  HIP_ASSERT(hipMemcpy(d_x, x, N*sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_y, y, N*sizeof(float), hipMemcpyHostToDevice));

  test( d_x, d_y, x, y, N);
  test2( d_x, d_y, x, y, N);

  // TODO: Add hipFree and free calls for proper cleanup

  return 0; // Added return statement
} 