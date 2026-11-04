// https://github.com/CHIP-SPV/chipStar/issues/2280: the timeout path must not block on a stream callback that needs a lock it holds.
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <atomic>
#include <cstdio>
#include <thread>
#include <unistd.h>

#define REQUIRE(C) if (!(C)) { printf("FAIL: %s\n", #C); fflush(stdout); _exit(1); }

static hipStream_t S, S2;
static void *A, *B;
static std::atomic<bool> Go{false}, Cb2Ran{false}, Cb3Ran{false};

static void Cb3(hipStream_t, hipError_t, void *) {
  sleep(1);
  REQUIRE(hipMemcpyAsync(B, A, 64, hipMemcpyDeviceToDevice, S2) == hipSuccess);
  Cb3Ran = true;
}

static void Cb(hipStream_t, hipError_t, void *) {
  while (!Go) usleep(1000);
  sleep(10); // outlast the 5 s thread wait, then need chipStar locks
  REQUIRE(hipMemcpyAsync(B, A, 64, hipMemcpyDeviceToDevice, S2) == hipSuccess);
  REQUIRE(hipStreamAddCallback(S, Cb3, nullptr, 0) == hipSuccess);
  REQUIRE(hipMemcpyAsync(B, A, 64, hipMemcpyDeviceToDevice, S) == hipSuccess);
}

static void Cb2(hipStream_t, hipError_t, void *) { Cb2Ran = true; }

int main() {
  REQUIRE(hipStreamCreateWithFlags(&S, hipStreamNonBlocking) == hipSuccess);
  REQUIRE(hipStreamCreateWithFlags(&S2, hipStreamNonBlocking) == hipSuccess);
  REQUIRE(hipMalloc(&A, 64) == hipSuccess);
  REQUIRE(hipMalloc(&B, 64) == hipSuccess);
  REQUIRE(hipStreamAddCallback(S, Cb, nullptr, 0) == hipSuccess);
  REQUIRE(hipStreamAddCallback(S, Cb2, nullptr, 0) == hipSuccess);
  REQUIRE(hipMemcpyAsync(B, A, 64, hipMemcpyDeviceToDevice, S) == hipSuccess);
  std::atomic<bool> Up{false};
  std::thread([&] { int N; REQUIRE(hipGetDeviceCount(&N) == hipSuccess); Up = true; for (;;) pause(); }).detach();
  while (!Up) usleep(1000);
  alarm(30); // SIGALRM kills the process if the wait never ends
  uintptr_t H[4] = {};
  Go = true;
  REQUIRE(hipInitFromNativeHandles(H, 4) == hipErrorNotSupported);
  REQUIRE(Cb2Ran);
  REQUIRE(Cb3Ran);
  printf("PASS\n");
  fflush(stdout);
  _exit(0);
}
