// This test tries to catch memory leak occuring during runtime by
// monitoring changes in process' resident memory size. Note: getting
// a pass with this test does not mean the absence of memory leaks.
#include <hip/hip_runtime.h>
#include <sys/resource.h>

__global__ void k() {}

size_t getMaxRSS() {
  rusage Rusage;
  if (getrusage(RUSAGE_SELF, &Rusage) != 0)
    exit(3);
  return Rusage.ru_maxrss;
}

int main() {
  // If set too low, the changes in maximum resident memory size might
  // be missed due to runtime's initial memory pools.
  constexpr size_t NumKernelLaunches = 128 * 1024;

  // A threshold measured in number of consecutive kernel launches
  // without an increase in the maximum resident set size (MaxRSS)
  // used as indicator for possible memory leakage.
  constexpr size_t NoLeakThreshold = NumKernelLaunches / 8u;

  size_t PrevMaxRSS = 0;
  size_t RSSIncLag = 0; // Number of Kernel launches since the last
                        // time the MaxRSS value increased.
  for (size_t i = 0; i < NumKernelLaunches; i++) {
    k<<<1, 1>>>();
    if (hipGetLastError() != hipSuccess)
      return 2;
    hipDeviceSynchronize();

    size_t CurrMaxRSS = getMaxRSS();
    if (CurrMaxRSS > PrevMaxRSS)
      RSSIncLag = 0;
    else
      RSSIncLag++;
    PrevMaxRSS = CurrMaxRSS;
  }

  printf("RSSIncLag=%zu\n", RSSIncLag);

  if (RSSIncLag < NoLeakThreshold) {
    printf("Possible memory leak!\n");
    return 1;
  }

  printf("Didn't detect memory leak.\n");
  return 0;
}
