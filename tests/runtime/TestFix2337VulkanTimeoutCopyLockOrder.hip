// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2337:
// hipInitFromNativeHandles while another thread is in hipMemcpyAsync must not
// deadlock on a queue's CmdRecordMtx_ and EventsMtx.
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <thread>
#include <unistd.h>

static thread_local bool Stall = false;
static std::atomic<bool> Stalled{false};

// Parks the copy's thread under CmdRecordMtx_, before EventsMtx.
extern "C" int vkEndCommandBuffer(void *Cb) {
  static auto Real = (int (*)(void *))dlsym(RTLD_NEXT, "vkEndCommandBuffer");
  if (Stall) {
    Stalled = true;
    for (;;)
      pause(); // never releases CmdRecordMtx_; alarm(40) bounds a deadlock
  }
  return Real(Cb);
}

#define REQUIRE(C)                                                             \
  if (!(C)) {                                                                  \
    printf("FAIL: %s\n", #C);                                                  \
    fflush(stdout);                                                            \
    _exit(1);                                                                  \
  }

__global__ void kernel() { printf("hello\n"); }

int main() {
  alarm(40); // SIGALRM kills the process if a deadlock occurs
  hipStream_t S;
  REQUIRE(hipStreamCreateWithFlags(&S, hipStreamNonBlocking) == hipSuccess);
  void *Src, *Dst;
  REQUIRE(hipMalloc(&Src, 64) == hipSuccess);
  REQUIRE(hipMalloc(&Dst, 64) == hipSuccess);
  kernel<<<1, 1, 0, S>>>();
  REQUIRE(hipStreamSynchronize(S) == hipSuccess);
  std::thread([=] {
    Stall = true;
    hipMemcpyAsync(Dst, Src, 64, hipMemcpyDeviceToDevice, S);
  }).detach();
  while (!Stalled)
    usleep(1000);
  uintptr_t H[4] = {};
  REQUIRE(hipInitFromNativeHandles(H, 4) == hipErrorNotSupported);
  printf("PASS\n");
  fflush(stdout);
  _exit(0); // skips teardown; the parked thread holds a deleted queue's lock
}
