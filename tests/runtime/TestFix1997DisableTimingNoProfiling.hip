// #1997: recording a hipEventDisableTiming event must keep the regular queue.

#include <hip/hip_runtime.h>
#include "backend/OpenCL/CHIPBackendOpenCL.hh"

#include <cstdio>

static bool isProfiling(hipStream_t Stream) {
  auto *Q = static_cast<CHIPQueueOpenCL *>(
      Backend->findQueue(reinterpret_cast<chipstar::Queue *>(Stream)));
  return Q->get()->getInfo<CL_QUEUE_PROPERTIES>() & CL_QUEUE_PROFILING_ENABLE;
}

int main() {
  hipStream_t Stream;
  hipEvent_t NoTiming, Timing;
  if (hipStreamCreate(&Stream) != hipSuccess ||
      hipEventCreateWithFlags(&NoTiming, hipEventDisableTiming) != hipSuccess ||
      hipEventCreate(&Timing) != hipSuccess)
    return 1;
  if (!dynamic_cast<CHIPBackendOpenCL *>(Backend)) {
    printf("HIP_SKIP_THIS_TEST: not running on OpenCL\n");
    return CHIP_SKIP_TEST;
  }

  if (hipEventRecord(NoTiming, Stream) != hipSuccess)
    return 1;
  if (isProfiling(Stream)) {
    printf("FAIL: a hipEventDisableTiming record switched to profiling\n");
    return 1;
  }

  // A timing event must still switch, or the check above proves nothing.
  if (hipEventRecord(Timing, Stream) != hipSuccess ||
      hipEventSynchronize(Timing) != hipSuccess)
    return 1;
  if (!isProfiling(Stream)) {
    printf("HIP_SKIP_THIS_TEST: queue profiling is unavailable\n");
    return CHIP_SKIP_TEST;
  }

  printf("PASS\n");
  return 0;
}
