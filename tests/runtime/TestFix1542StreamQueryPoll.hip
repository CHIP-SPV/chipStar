// hipStreamQuery on the OpenCL backend, on an implementation that submits
// lazily (Arm Mali). https://github.com/CHIP-SPV/chipStar/issues/1542
//
// Five properties, each of which fails on a different piece of
// CHIPQueueOpenCL::query. Phases 1 to 3 separate implementations only on one
// that submits lazily: Intel and pocl submit eagerly and complete an idle
// queue's marker in the call that enqueued it, so those three pass there
// whichever way query polls. Phase 4 is about a stale marker rather than
// about submission and can fail anywhere. The watchdog turns a hang into a
// FAIL so CI does not stall.
//
//  1. A poll loop after kernel launches reaches hipSuccess at all. query has
//     to flush the queue, because clGetEventInfo does not, and it has to poll
//     one marker across calls, because a marker read in the call that
//     enqueued it is CL_QUEUED here. Run on the stream's regular command
//     queue and again on its profiling command queue, which hipEventRecord
//     switches it to.
//  2. Once a poll has reported hipSuccess, further polls of that stream keep
//     reporting it while nothing is enqueued. query has to record the queue
//     it just proved drained; otherwise the next call asks a fresh marker and
//     answers hipErrorNotReady on every other call.
//  3. A poll loop on one stream finishes while another stream keeps
//     launching. Enqueueing on one stream puts a marker on the others to
//     order them; that marker carries no work and must not tear down the
//     marker their polls are on, or the poll starves.
//  4. A poll does not report a stream drained while work enqueued after the
//     polled marker is still running. This is what every dropQueryMarker call
//     site buys, and it is the failure the other three cannot show: a wrong
//     answer rather than a missing one.
//  5. A stream whose cl_command_queue has been handed to the application is
//     still answerable. query() cannot see what the application enqueues
//     there, so it stops keeping a marker for such a stream and may report it
//     not ready for as long as the implementation defers submission; what it
//     must not do is fail, abort or hang.

#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <unistd.h>

#define CHECK(Expr)                                                            \
  do {                                                                         \
    hipError_t Err = (Expr);                                                   \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %s\n", #Expr, hipGetErrorString(Err));         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void bump(int *P) { *P += 1; }

/// Runs long enough to be observable from the host. Volatile so the loop
/// survives the optimizer, unsigned so the accumulation cannot overflow into
/// undefined behavior.
__global__ void spin(int *P, unsigned Iters) {
  volatile unsigned Acc = 0;
  for (unsigned I = 0; I < Iters; ++I)
    Acc += I;
  *P += (Acc == 0u);
}

static const char *const PhaseMsg[] = {
    "FAIL: watchdog: hipStreamQuery never reported completion\n",
    "FAIL: watchdog: hipStreamQuery never reported completion after "
    "hipEventRecord\n",
    "FAIL: watchdog: hipStreamQuery never reported completion while another "
    "stream was launching\n",
    "FAIL: watchdog: hipStreamQuery never reported completion behind a "
    "long kernel\n",
};
static volatile sig_atomic_t Phase = 0;

static void watchdog(int) {
  const char *Msg = PhaseMsg[Phase];
  size_t Len = 0;
  while (Msg[Len])
    ++Len;
  (void)!write(STDOUT_FILENO, Msg, Len);
  _exit(1);
}

static int pollUntilDone(hipStream_t S) {
  for (;;) {
    hipError_t Err = hipStreamQuery(S);
    if (Err == hipSuccess)
      return 0;
    if (Err != hipErrorNotReady) {
      printf("FAIL: hipStreamQuery returned %s\n", hipGetErrorString(Err));
      return 1;
    }
    usleep(1000);
  }
}

using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point Start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - Start).count();
}

int main() {
  signal(SIGALRM, watchdog);
  alarm(120);

  bool Skipped4 = false;
  int *P = nullptr;
  CHECK(hipMalloc(&P, sizeof(int)));
  CHECK(hipMemset(P, 0, sizeof(int)));

  const int Rounds = 10;

  // 1a. Launches on the regular queue, then poll without anything that flushes.
  Phase = 0;
  for (int I = 0; I < Rounds; ++I) {
    bump<<<1, 1>>>(P);
    bump<<<1, 1>>>(P);
    if (pollUntilDone(0))
      return 1;
  }

  // 2. The stream is drained and nothing is enqueued, so every poll must say
  // so. One that asks the driver a fresh marker each time answers
  // hipErrorNotReady on every other call here.
  for (int I = 0; I < 20; ++I) {
    hipError_t Err = hipStreamQuery(0);
    if (Err != hipSuccess) {
      printf("FAIL: hipStreamQuery on a drained stream returned %s on poll "
             "%d\n",
             hipGetErrorString(Err), I);
      return 1;
    }
    usleep(2000);
  }

  // 1b. A timing event moves the stream to its profiling queue; the launches
  // behind it and the polled marker now sit on that queue.
  hipEvent_t E;
  CHECK(hipEventCreate(&E));
  CHECK(hipEventRecord(E, 0));
  Phase = 1;
  for (int I = 0; I < Rounds; ++I) {
    bump<<<1, 1>>>(P);
    bump<<<1, 1>>>(P);
    if (pollUntilDone(0))
      return 1;
  }

  const int Expected = 4 * Rounds;
  int Count = -1;
  CHECK(hipMemcpy(&Count, P, sizeof(int), hipMemcpyDeviceToHost));
  if (Count != Expected) {
    printf("FAIL: counter %d, expected %d\n", Count, Expected);
    return 1;
  }

  // 3. Poll one stream while another keeps launching. Both directions: a
  // launch on the null stream orders itself against every blocking stream,
  // and a launch on a blocking stream orders itself against the null stream,
  // so each puts a marker on the queue the other one is polling.
  hipStream_t S;
  CHECK(hipStreamCreate(&S));
  Phase = 2;
  int *Q = nullptr;
  CHECK(hipMalloc(&Q, sizeof(int)));
  CHECK(hipMemset(Q, 0, sizeof(int)));

  // Bounded rather than left to the watchdog: a starved poll would otherwise
  // enqueue launches for the whole timeout.
  const int MaxPolls = 2000;

  bump<<<1, 1>>>(P);
  int Polls = 0;
  for (;; ++Polls) {
    if (Polls == MaxPolls) {
      printf("FAIL: hipStreamQuery(0) never reported completion in %d polls "
             "while another stream was launching\n",
             MaxPolls);
      return 1;
    }
    bump<<<1, 1, 0, S>>>(Q); // traffic on the other stream
    hipError_t Err = hipStreamQuery(0);
    if (Err == hipSuccess)
      break;
    if (Err != hipErrorNotReady) {
      printf("FAIL: hipStreamQuery(0) returned %s\n", hipGetErrorString(Err));
      return 1;
    }
    usleep(1000);
  }
  CHECK(hipStreamSynchronize(0));
  CHECK(hipStreamSynchronize(S));

  bump<<<1, 1, 0, S>>>(Q);
  for (Polls = 0;; ++Polls) {
    if (Polls == MaxPolls) {
      printf("FAIL: hipStreamQuery(S) never reported completion in %d polls "
             "while the null stream was launching\n",
             MaxPolls);
      return 1;
    }
    bump<<<1, 1>>>(P); // traffic on the null stream
    hipError_t Err = hipStreamQuery(S);
    if (Err == hipSuccess)
      break;
    if (Err != hipErrorNotReady) {
      printf("FAIL: hipStreamQuery(S) returned %s\n", hipGetErrorString(Err));
      return 1;
    }
    usleep(1000);
  }
  CHECK(hipStreamSynchronize(0));
  CHECK(hipStreamSynchronize(S));

  // 4. Work enqueued after the polled marker must hold the poll off. Time one
  // spin kernel first: a stream whose poll ignores the second launch reports
  // itself drained as soon as the first one ends, leaving nearly a whole
  // kernel to run inside the hipStreamSynchronize that follows.
  Phase = 3;
  unsigned Iters = 1u << 20;
  double Kernel = 0.0;
  for (int Attempt = 0; Attempt < 8; ++Attempt) {
    spin<<<1, 1, 0, S>>>(Q, Iters); // warm up the launch path
    CHECK(hipStreamSynchronize(S));
    auto Start = Clock::now();
    spin<<<1, 1, 0, S>>>(Q, Iters);
    CHECK(hipStreamSynchronize(S));
    Kernel = msSince(Start);
    if (Kernel >= 40.0)
      break; // Iters is the count Kernel was measured at
    Iters *= 2;
  }

  if (Kernel < 40.0) {
    printf("note: phase 4 skipped, a %d iteration kernel still runs in only "
           "%.1f ms\n",
           Iters, Kernel);
    Skipped4 = true;
  } else {
    spin<<<1, 1, 0, S>>>(Q, Iters);
    hipError_t Err = hipStreamQuery(S); // caches a marker covering this launch
    if (Err != hipErrorNotReady && Err != hipSuccess) {
      printf("FAIL: hipStreamQuery(S) returned %s\n", hipGetErrorString(Err));
      return 1;
    }
    spin<<<1, 1, 0, S>>>(Q, Iters); // must invalidate that marker
    if (pollUntilDone(S))
      return 1;
    auto Start = Clock::now();
    CHECK(hipStreamSynchronize(S));
    double Left = msSince(Start);
    if (Left > Kernel / 2) {
      printf("FAIL: hipStreamQuery reported the stream drained with %.1f ms of "
             "a %.1f ms kernel still to run\n",
             Left, Kernel);
      return 1;
    }
  }

  // 5. Hand the stream's native queue to the application, then keep querying
  // it. query() switches to a marker enqueued in the same call, which on an
  // implementation that defers submission may never report ready, so this
  // asks only that the call keeps answering with a legal status.
  uintptr_t NativeHandles[8];
  int NumHandles = 0;
  if (hipGetBackendNativeHandles((uintptr_t)S, nullptr, &NumHandles) !=
      hipSuccess) {
    printf("FAIL: hipGetBackendNativeHandles could not report a handle "
           "count\n");
    return 1;
  }
  if (NumHandles > (int)(sizeof(NativeHandles) / sizeof(NativeHandles[0]))) {
    printf("FAIL: hipGetBackendNativeHandles wants %d handles\n", NumHandles);
    return 1;
  }
  // Exactly one of the two pointers may be non null: the first call asks the
  // count, the second fills the array.
  if (hipGetBackendNativeHandles((uintptr_t)S, NativeHandles, nullptr) !=
      hipSuccess) {
    printf("FAIL: hipGetBackendNativeHandles failed\n");
    return 1;
  }
  bump<<<1, 1, 0, S>>>(Q);
  for (int I = 0; I < 200; ++I) {
    hipError_t Err = hipStreamQuery(S);
    if (Err != hipSuccess && Err != hipErrorNotReady) {
      printf("FAIL: hipStreamQuery on a stream with an exported native queue "
             "returned %s\n",
             hipGetErrorString(Err));
      return 1;
    }
    usleep(1000);
  }
  CHECK(hipStreamSynchronize(S));

  CHECK(hipStreamDestroy(S));
  CHECK(hipEventDestroy(E));
  CHECK(hipFree(Q));
  CHECK(hipFree(P));
  printf("counter check done, phase 4 %s\n", Skipped4 ? "skipped" : "ran");
  printf("PASS\n");
  return 0;
}
