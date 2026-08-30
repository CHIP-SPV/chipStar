// Reproduces #1543: CHIPQueueOpenCL::switchModeTo leaks the barrier event it
// enqueues on the queue being switched to, once per stream that switches to
// the profiling queue.
//
// The first hipEventRecord on a stream is what triggers the switch, so this
// program only has to record an event on one stream, wait for it and tear
// everything down. The leak is invisible through the HIP API: an LD_PRELOAD
// interposer counts the reference balance of every cl_event that
// clEnqueueBarrierWithWaitList returned and reports it at process exit.
// tests/run_barrier_event_leak_test.cmake drives the run and owns the verdict.

#include <hip/hip_runtime.h>

#include <cstdio>

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::printf("%s failed: %s\n", #Call, hipGetErrorString(Err));           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void touch(int *Out) { Out[threadIdx.x] = threadIdx.x; }

int main() {
  int *Buf = nullptr;
  hipStream_t Stream = nullptr;
  hipEvent_t Start = nullptr;
  hipEvent_t Stop = nullptr;
  CHECK(hipMalloc(&Buf, 64 * sizeof(int)));
  CHECK(hipStreamCreate(&Stream));
  CHECK(hipEventCreate(&Start));
  CHECK(hipEventCreate(&Stop));

  // Switches Stream to its profiling queue.
  CHECK(hipEventRecord(Start, Stream));
  hipLaunchKernelGGL(touch, dim3(1), dim3(64), 0, Stream, Buf);
  CHECK(hipEventRecord(Stop, Stream));
  CHECK(hipEventSynchronize(Stop));

  float Ms = 0.0f;
  CHECK(hipEventElapsedTime(&Ms, Start, Stop));

  CHECK(hipEventDestroy(Stop));
  CHECK(hipEventDestroy(Start));
  CHECK(hipStreamDestroy(Stream));
  CHECK(hipFree(Buf));

  std::printf("recorded, elapsed %f ms\n", Ms);
  std::fflush(stdout);
  return 0;
}
