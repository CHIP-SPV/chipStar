// Reproduces: on the OpenCL backend hipEventSynchronize never returns when
// hipEventRecord switched the stream from its regular command queue to its
// profiling queue while the regular queue still held unsubmitted work.
// https://github.com/CHIP-SPV/chipStar/issues/1538
//
// The switch enqueues a barrier on the profiling queue that waits on a marker
// of the regular queue. clWaitForEvents flushes only the queue of the event it
// waits on, so an OpenCL implementation that submits lazily (Mali) never sees
// the marker and the barrier waits forever. Implementations that submit
// eagerly (Intel, pocl) hide the missing flush, so this passes there with or
// without the fix. The watchdog turns a hang into a FAIL so CI does not stall.
//
// Three ways of waiting for an event recorded after unsubmitted work are
// covered: hipEventSynchronize, polling hipEventQuery, and hipEventSynchronize
// on a stream that first waited on an event of another stream.

#include <hip/hip_runtime.h>
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

static const char *const PhaseMsg[] = {
    "FAIL: watchdog: hipEventSynchronize hung\n",
    "FAIL: watchdog: hipEventQuery never reported completion\n",
    "FAIL: watchdog: hipEventSynchronize hung after hipStreamWaitEvent\n",
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

int main() {
  signal(SIGALRM, watchdog);
  alarm(60);

  int *P = nullptr;
  CHECK(hipMalloc(&P, sizeof(int)));
  CHECK(hipMemset(P, 0, sizeof(int)));

  // Work on the regular queue, then a timing event moves the stream to the
  // profiling queue and the wait depends on the never flushed regular queue.
  hipEvent_t E0;
  CHECK(hipEventCreate(&E0));
  bump<<<1, 1>>>(P);
  bump<<<1, 1>>>(P);
  CHECK(hipEventRecord(E0, 0));
  Phase = 0;
  CHECK(hipEventSynchronize(E0));

  // Polling has no implicit flush at all.
  hipEvent_t E1;
  CHECK(hipEventCreate(&E1));
  bump<<<1, 1>>>(P);
  CHECK(hipEventRecord(E1, 0));
  Phase = 1;
  for (;;) {
    hipError_t Err = hipEventQuery(E1);
    if (Err == hipSuccess)
      break;
    if (Err != hipErrorNotReady) {
      printf("FAIL: hipEventQuery returned %s\n", hipGetErrorString(Err));
      return 1;
    }
    usleep(1000);
  }

  // An event of another stream that the null stream waits on: the barrier on
  // the null stream depends on the other stream's queue.
  hipStream_t S;
  hipEvent_t E2, E3;
  CHECK(hipStreamCreateWithFlags(&S, hipStreamNonBlocking));
  CHECK(hipEventCreate(&E2));
  CHECK(hipEventCreate(&E3));
  bump<<<1, 1, 0, S>>>(P);
  CHECK(hipEventRecord(E2, S));
  CHECK(hipStreamWaitEvent(0, E2, 0));
  bump<<<1, 1>>>(P);
  CHECK(hipEventRecord(E3, 0));
  Phase = 2;
  CHECK(hipEventSynchronize(E3));

  int Count = -1;
  CHECK(hipMemcpy(&Count, P, sizeof(int), hipMemcpyDeviceToHost));

  CHECK(hipStreamDestroy(S));
  CHECK(hipEventDestroy(E0));
  CHECK(hipEventDestroy(E1));
  CHECK(hipEventDestroy(E2));
  CHECK(hipEventDestroy(E3));
  CHECK(hipFree(P));

  bool Pass = Count == 5;
  printf("counter %d (expected 5): %s\n", Count, Pass ? "PASS" : "FAIL");
  return Pass ? 0 : 1;
}
