// Check indirect global buffer access (IGBA) detection.
#include <hip/hip_runtime.h>
#include "CHIPBackend.hh"

struct SomeStruct {
  int Bar;
  int *Ptr;
};

__global__ void k(SomeStruct Args) { *Args.Ptr = 123; }

int main() {
  auto *ChipKernel = Backend->getActiveDevice()->findKernel(
      HostPtr(reinterpret_cast<const void *>(k)));
  if (ChipKernel->getModule()->getInfo().MayHaveIGBAs)
    return 0;
  return 1;
}
