// Reproduces https://github.com/CHIP-SPV/chipStar/issues/1556: a kernel launch
// from a module that can abort must not wait for the kernel to complete.
//
// A host function enqueued ahead of the kernel waits for the launch to return,
// so a launch that waits for the kernel cannot return until the host function
// gives up after TimeoutSeconds, and the test then fails instead of hanging.

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

constexpr int TimeoutSeconds = 10;

static std::atomic<bool> HostFuncEnqueued{false}, LaunchReturned{false},
    TimedOut{false}, HostFuncSynchronous{false};

static bool waitFor(std::atomic<bool> &Flag, std::chrono::seconds Timeout) {
  auto Deadline = std::chrono::steady_clock::now() + Timeout;
  while (!Flag) {
    if (std::chrono::steady_clock::now() > Deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

static void waitForLaunch(void *) {
  // The OpenCL backend runs the host function before hipLaunchHostFunc returns.
  if (!waitFor(HostFuncEnqueued, std::chrono::seconds(TimeoutSeconds))) {
    HostFuncSynchronous = true;
    return;
  }
  TimedOut = !waitFor(LaunchReturned, std::chrono::seconds(TimeoutSeconds));
}

__global__ void mayAbort(int *Out) {
  if (*Out == 2) // Never taken; keeps the abort flag in the module.
    abort();
  *Out = 1;
}

int main() {
  int *Out;
  if (hipMalloc(&Out, sizeof(int)) != hipSuccess ||
      hipMemset(Out, 0, sizeof(int)) != hipSuccess) {
    std::printf("FAIL: allocation\n");
    return 1;
  }
  hipStream_t S;
  if (hipStreamCreate(&S) != hipSuccess) {
    std::printf("FAIL: hipStreamCreate\n");
    return 1;
  }

  // The first launch initializes device variables, which synchronizes.
  mayAbort<<<1, 1, 0, S>>>(Out);
  if (hipStreamSynchronize(S) != hipSuccess) {
    std::printf("FAIL: warm-up launch\n");
    return 1;
  }

  if (hipLaunchHostFunc(S, waitForLaunch, nullptr) != hipSuccess) {
    std::printf("FAIL: hipLaunchHostFunc\n");
    return 1;
  }
  HostFuncEnqueued = true;
  mayAbort<<<1, 1, 0, S>>>(Out);
  LaunchReturned = true;

  int Result = 0;
  if (hipStreamSynchronize(S) != hipSuccess ||
      hipMemcpy(&Result, Out, sizeof(int), hipMemcpyDeviceToHost) !=
          hipSuccess) {
    std::printf("FAIL: synchronize\n");
    return 1;
  }
  if (HostFuncSynchronous) {
    std::printf("HIP_SKIP_THIS_TEST: hipLaunchHostFunc waits for the host "
                "function\n");
    return CHIP_SKIP_TEST;
  }
  if (TimedOut) {
    std::printf("FAIL: the launch blocked until the kernel finished\n");
    return 1;
  }
  if (Result != 1) {
    std::printf("FAIL: kernel result %d\n", Result);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
