// A static table holding dynamic shared memory addresses hung hipcc (#1692);
// it cannot be lowered, so hipcc must say so instead.
#include <hip/hip_runtime.h>
__global__ void k(size_t *Out, unsigned C) {
  extern __shared__ float Dyn[];
  static float *const P[2] = {Dyn, Dyn + 16};
  Out[0] = (size_t)P[C & 1];
}
