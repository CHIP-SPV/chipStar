// Reproduces https://github.com/CHIP-SPV/chipStar/issues/1857: the launch
// entry points declared extern "C" by hip/hip_ext.h do not link against
// libCHIP.

#include <hip/hip_ext.h>
#include <cstdio>

__global__ void k(int *Out) { *Out = 42; }

int main() {
  int *Out;
  int Host = 0;
  hipEvent_t Start, Stop;
  if (hipMalloc(&Out, sizeof(int)) != hipSuccess ||
      hipEventCreate(&Start) != hipSuccess ||
      hipEventCreate(&Stop) != hipSuccess) {
    printf("FAILED: setup\n");
    return 1;
  }
  void *Args[] = {&Out};
  float Ms = 0;
  if (hipExtLaunchKernel(reinterpret_cast<void *>(k), dim3(1), dim3(1), Args,
                         0, nullptr, Start, Stop, 0) != hipSuccess ||
      hipMemcpy(&Host, Out, sizeof(int), hipMemcpyDeviceToHost) !=
          hipSuccess ||
      Host != 42 || hipEventElapsedTime(&Ms, Start, Stop) != hipSuccess) {
    printf("FAILED: hipExtLaunchKernel, Out = %d\n", Host);
    return 1;
  }

  // A null kernel is rejected; the call only has to link and return.
  if (hipExtModuleLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr,
                               nullptr) == hipSuccess ||
      hipHccModuleLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr,
                               nullptr) == hipSuccess) {
    printf("FAILED: null kernel accepted\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
