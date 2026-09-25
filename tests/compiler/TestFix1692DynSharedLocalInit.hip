// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1692: a local
// array initialized with dynamic shared memory addresses hung or aborted hipcc.
#include <hip/hip_runtime.h>

// Two uses of Dyn in the initializer: the hang in #1692.
__global__ void k2(size_t *Out, unsigned C) {
  extern __shared__ float Dyn[];
  float *P[2] = {Dyn, Dyn + 16};
  Out[0] = (size_t)P[C & 1];
}

// Part of the table copied out: optimized, it is read from the initializer.
__global__ void k3(size_t *Out, unsigned C) {
  extern __shared__ float Dyn[];
  float *P[4] = {Dyn, Dyn + 16, Dyn + 32, Dyn + 48}, *Q[2];
  __builtin_memcpy(Q, P + 1, sizeof(Q));
  Out[0] = (size_t)Q[C & 1];
}

// One use of Dyn1 in the initializer: the abort in #1692.
__global__ void k1(size_t *Out, unsigned C) {
  extern __shared__ int Dyn1[];
  int *P[2] = {Dyn1, nullptr};
  Out[0] = (size_t)P[C & 1];
}
