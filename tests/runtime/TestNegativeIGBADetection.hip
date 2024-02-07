// Check we detect that the device code does not have indirect global
// buffer accesses.
#include <hip/hip_runtime.h>
#include "CHIPBackend.hh"

__global__ void k(int *Out) { *Out = 123; }

int main() {
  auto *ChipKernel = Backend->getActiveDevice()->findKernel(
      HostPtr(reinterpret_cast<const void *>(k)));
  if (!ChipKernel->getModule()->getInfo().MayHaveIGBAs)
    return 0;
  return 1;
}
