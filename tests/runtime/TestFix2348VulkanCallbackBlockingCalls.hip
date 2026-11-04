// Blocking HIP calls made inside a stream callback must not wait on the
// callback itself. https://github.com/CHIP-SPV/chipStar/issues/2348
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

static constexpr int N = 256;
static hipStream_t S, T;
static hipEvent_t E;
static int *DevA, *DevB, *DevC, *Pat, *H, *HostPinned;
static int Expect = 0, Seq = 0;
static std::atomic<int> Done{0};
static std::atomic<bool> Go{false};
static bool Bad;

__global__ void Inc(int *A) { A[0] += 1; }

#define OK(x) do { if ((x) != hipSuccess) Bad = true; } while (0)
#define KERNEL(Strm, Ptr) hipLaunchKernelGGL(Inc, dim3(1), dim3(1), 0, Strm, Ptr)

typedef void (*Fn)(hipStream_t, hipError_t, void *);
static void Tramp(hipStream_t St, hipError_t Er, void *F) {
  ((Fn)F)(St, Er, nullptr);
  ++Done;
}

// Adds Cb on S, runs After on main, and waits for Want callbacks to finish.
static void run(Fn Cb, void (*After)() = nullptr, int Want = 1) {
  Done = 0;
  Seq = 0;
  hipStreamAddCallback(S, Tramp, (void *)Cb, 0);
  if (After)
    After();
  while (Done < Want)
    usleep(1000);
  OK(hipStreamSynchronize(S));
}

static void d2h(hipStream_t, hipError_t, void *) {
  OK(hipMemcpyAsync(H, DevA, N * 4, hipMemcpyDeviceToHost, S));
  Seq = Seq * 10 + 1;
}
static void second(hipStream_t, hipError_t, void *) { Seq = Seq * 10 + 2; }
static void h2h(hipStream_t, hipError_t, void *) {
  OK(hipMemcpyAsync(H, Pat, N * 4, hipMemcpyDefault, S));
}
static void fill(hipStream_t, hipError_t, void *) {
  OK(hipMemsetAsync(HostPinned, 0x5A, N * 4, S));
}
static void record2(hipStream_t, hipError_t, void *) {
  OK(hipEventRecord(E, S));
  OK(hipEventRecord(E, S));
}
static void sync(hipStream_t, hipError_t, void *) {
  OK(hipEventRecord(E, S));
  OK(hipEventSynchronize(E));
}
static void destroy(hipStream_t, hipError_t, void *) {
  OK(hipEventRecord(E, S));
  OK(hipEventDestroy(E));
}
static void waitT(hipStream_t, hipError_t, void *) {
  OK(hipEventRecord(E, S));
  OK(hipStreamWaitEvent(T, E, 0));
}
static void waitS(hipStream_t, hipError_t, void *) {
  OK(hipEventRecord(E, S));
  OK(hipStreamWaitEvent(S, E, 0));
}
static void gated(hipStream_t, hipError_t, void *) {
  while (!Go)
    usleep(1000);
  OK(hipMemcpyAsync(H, DevB, N * 4, hipMemcpyDeviceToHost, T));
}
static void launchAfterAdd() {
  KERNEL(T, DevB);
  KERNEL(S, DevC);
  ++Expect;
  Go = true;
}
static void addSecond() { hipStreamAddCallback(S, Tramp, (void *)second, 0); }
static void pool(hipStream_t, hipError_t, void *) {
  for (int I = 0; I < 10; ++I)
    KERNEL(S, DevC);
}

int main() {
  alarm(120);
  Pat = (int *)malloc(N * 4);
  H = (int *)calloc(N, 4);
  for (int I = 0; I < N; ++I)
    Pat[I] = I * 3 + 1;
  OK(hipStreamCreateWithFlags(&S, hipStreamNonBlocking));
  OK(hipStreamCreateWithFlags(&T, hipStreamNonBlocking));
  OK(hipEventCreate(&E));
  OK(hipMalloc(&DevA, N * 4));
  OK(hipMalloc(&DevB, N * 4));
  OK(hipMalloc(&DevC, N * 4));
  OK(hipHostMalloc((void **)&HostPinned, N * 4));
  OK(hipMemcpy(DevA, Pat, N * 4, hipMemcpyHostToDevice));
  OK(hipMemset(DevB, 0, N * 4));
  OK(hipMemset(DevC, 0, N * 4));
  KERNEL(S, DevC); // warm up the kernel off the monitor thread
  ++Expect;
  OK(hipDeviceSynchronize());

  run(d2h);
  Bad |= memcmp(H, Pat, N * 4) != 0;
  memset(H, 0, N * 4);
  run(h2h);
  Bad |= memcmp(H, Pat, N * 4) != 0;
  run(fill);
  for (int I = 0; I < N; ++I)
    Bad |= HostPinned[I] != 0x5A5A5A5A;
  run(record2);
  run(sync);
  run(destroy);
  OK(hipEventCreate(&E));
  run(waitT);
  run(waitS);
  run(gated, launchAfterAdd);
  run(d2h, addSecond, 2);
  Bad |= Seq != 12;
  for (int I = 0; I < 1015; ++I) {
    KERNEL(S, DevC);
    ++Expect;
    if (I % 50 == 49)
      OK(hipStreamSynchronize(S));
  }
  OK(hipStreamSynchronize(S));
  run(pool);
  Expect += 10;

  int Count = 0;
  OK(hipMemcpy(&Count, DevC, 4, hipMemcpyDeviceToHost));
  Bad |= Count != Expect;
  puts(Bad ? "FAIL" : "PASS");
  return Bad;
}
