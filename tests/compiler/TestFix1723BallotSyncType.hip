#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void k(uint64_t *Out, unsigned Mask) {
  Out[threadIdx.x] = __ballot_sync(Mask, threadIdx.x & 1);
}
