// Reproduces #1812: a launch that chipstar::Queue::launch rejects leaks its
// execution item, through both hipLaunchKernel and hipLaunchByPtr.
//
// Replacing the global operator new/delete applies to the whole process, so
// counting live allocations sees the items the runtime creates and frees.

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>

static std::atomic<long> LiveAllocs{0};

void *operator new(std::size_t Size) {
  void *Ptr = std::malloc(Size ? Size : 1);
  if (!Ptr)
    throw std::bad_alloc();
  ++LiveAllocs;
  return Ptr;
}
void operator delete(void *Ptr) noexcept {
  if (!Ptr)
    return;
  --LiveAllocs;
  std::free(Ptr);
}
void operator delete(void *Ptr, std::size_t) noexcept { operator delete(Ptr); }

__global__ void k() {}

static hipError_t launch(bool ByPtr, dim3 Block) {
  if (!ByPtr) {
    void *Args[1] = {nullptr};
    return hipLaunchKernel(reinterpret_cast<const void *>(k), dim3(1), Block,
                           Args, 0, nullptr);
  }
  hipError_t Err = hipConfigureCall(dim3(1), Block, 0, nullptr);
  if (Err != hipSuccess)
    return Err;
  return hipLaunchByPtr(reinterpret_cast<const void *>(k));
}

// Returns how many allocations N rejected launches left behind.
static long leakedBy(bool ByPtr, dim3 Block, int N) {
  long Before = LiveAllocs;
  for (int I = 0; I < N; I++)
    if (launch(ByPtr, Block) != hipErrorInvalidValue)
      return -1;
  (void)hipGetLastError();
  return LiveAllocs - Before;
}

int main() {
  hipDeviceProp_t Props;
  if (hipGetDeviceProperties(&Props, 0) != hipSuccess) {
    std::printf("FAIL: hipGetDeviceProperties\n");
    return 1;
  }
  dim3 TooBig(Props.maxThreadsPerBlock + 1);
  constexpr int N = 256;

  bool Ok = true;
  for (bool ByPtr : {false, true}) {
    const char *Api = ByPtr ? "hipLaunchByPtr" : "hipLaunchKernel";
    leakedBy(ByPtr, TooBig, 8); // Warm up: module build, caches.
    long Leaked = leakedBy(ByPtr, TooBig, N);
    std::printf("%s: %d rejected launches left %ld allocations\n", Api, N,
                Leaked);
    if (Leaked < 0 || Leaked >= N) {
      std::printf("FAIL: %s %s\n", Api,
                  Leaked < 0 ? "did not return hipErrorInvalidValue" : "leaks");
      Ok = false;
    }
  }
  std::fflush(stdout);
  if (!Ok)
    return 1;
  std::printf("PASS\n");
  return 0;
}
