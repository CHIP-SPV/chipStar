// Reproduces: on the OpenCL backend a hipStreamQuery polling loop after
// kernel launches never reports hipSuccess on an implementation that submits
// lazily (Mali). https://github.com/CHIP-SPV/chipStar/issues/1542
//
// CHIPQueueOpenCL::query enqueues a marker, reads its execution status with
// clGetEventInfo and releases it. clGetEventInfo is not one of the calls that
// flush a command queue, so nothing ever submits the launches or the marker,
// and even once flushed a marker read in the call that enqueued it is only
// complete on implementations that finish an idle queue's marker at enqueue
// time: Mali reports it CL_QUEUED, and the next poll asks a fresh one. Intel
// and pocl hide both, so this passes there with or without the fix. The
// watchdog turns a hang into a FAIL so CI does not stall.
//
// Each round launches twice and polls to completion; the rounds repeat so a
// poll that completes only by luck of timing still fails the test. Two states
// of the stream are covered: its regular command queue, and its profiling
// command queue after hipEventRecord switched it there, since the query polls
// only the queue that is active at the time.

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
    "FAIL: watchdog: hipStreamQuery never reported completion\n",
    "FAIL: watchdog: hipStreamQuery never reported completion after "
    "hipEventRecord\n",
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

int main() {
  signal(SIGALRM, watchdog);
  alarm(60);

  int *P = nullptr;
  CHECK(hipMalloc(&P, sizeof(int)));
  CHECK(hipMemset(P, 0, sizeof(int)));

  const int Rounds = 10;

  // Launches on the regular queue, then poll without anything that flushes.
  Phase = 0;
  for (int I = 0; I < Rounds; ++I) {
    bump<<<1, 1>>>(P);
    bump<<<1, 1>>>(P);
    if (pollUntilDone(0))
      return 1;
  }

  // A timing event moves the stream to its profiling queue; the launches
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

  int Count = -1;
  CHECK(hipMemcpy(&Count, P, sizeof(int), hipMemcpyDeviceToHost));

  CHECK(hipEventDestroy(E));
  CHECK(hipFree(P));

  const int Expected = 4 * Rounds;
  bool Pass = Count == Expected;
  printf("counter %d (expected %d): %s\n", Count, Expected,
         Pass ? "PASS" : "FAIL");
  return Pass ? 0 : 1;
}
