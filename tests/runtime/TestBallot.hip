#include "CHIPDriver.hh"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>

__global__ void k(uint64_t *Out, uint64_t In) {
  auto Votes = __ballot((In >> threadIdx.x) & 1u);
  if (threadIdx.x == 0)
    *Out = Votes;
}

void checkBallot(unsigned BlockSize, uint64_t In,
                 const hipDeviceProp_t &Props) {
  uint64_t *OutD, OutH = ~In;
  (void)hipMalloc(&OutD, sizeof(uint64_t));
  k<<<1, BlockSize>>>(OutD, In);
  (void)hipMemcpy(&OutH, OutD, sizeof(uint64_t), hipMemcpyDeviceToHost);

  if (Props.warpSize < 64 && OutH >> Props.warpSize) {
    printf("FAILED: Garbage bits in the __ballot result!\n");
    exit(1);
  }

  if (OutH != In) {
    printf("FAILED: BlockSize=%u:\nError: Expected '%lu'. Got '%lu'\n", BlockSize, In,
           OutH);
    exit(1);
  }
  (void)hipFree(OutD);
}

int main() {
  hipDeviceProp_t Props;
  (void)hipGetDeviceProperties(&Props, 0);
  if (!Props.arch.hasWarpBallot) {
    printf("SKIP: device does not support __ballot()\n");
    return CHIP_SKIP_TEST;
  }

  checkBallot(32, 0xBADF00D1, Props);
  std::cout << "PASSED\n";
  return 0;
}
