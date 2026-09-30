// __ballot_sync must return the vote bits under a full and a partial mask
// (CHIP-SPV/chipStar#1723).
#include "CHIPDriver.hh"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

__global__ void k(uint64_t *Out, unsigned Mask, uint32_t In) {
  uint64_t Votes = __ballot_sync(Mask, (In >> threadIdx.x) & 1u);
  if (threadIdx.x == 0)
    *Out = Votes;
}

static bool check(unsigned Mask, uint32_t In) {
  uint64_t *OutD, OutH = 0;
  (void)hipMalloc(&OutD, sizeof(uint64_t));
  k<<<1, 32>>>(OutD, Mask, In);
  (void)hipMemcpy(&OutH, OutD, sizeof(uint64_t), hipMemcpyDeviceToHost);
  (void)hipFree(OutD);
  uint64_t Expected = In & Mask;
  if (OutH != Expected) {
    printf("FAILED: Mask=0x%x: expected 0x%llx, got 0x%llx\n", Mask,
           (unsigned long long)Expected, (unsigned long long)OutH);
    return false;
  }
  return true;
}

int main() {
  hipDeviceProp_t Props;
  (void)hipGetDeviceProperties(&Props, 0);
  if (!Props.arch.hasWarpBallot || Props.warpSize != 32) {
    printf("HIP_SKIP_THIS_TEST: needs __ballot_sync with a 32-wide warp\n");
    return CHIP_SKIP_TEST;
  }
  if (!check(0xFFFFFFFFu, 0xBADF00D1u) || !check(0x0000FFFFu, 0xBADF00D1u))
    return 1;
  printf("PASSED\n");
  return 0;
}
