// Legacy stream work and hipDeviceSynchronize wait for another thread's
// hipStreamPerThread.

#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdio>
#include <thread>

#define N (1024 * 1024 * 8)

__global__ void slowFill(int *P) {
  for (int I = 0; I < N; ++I)
    P[I] = I + 1;
}

static long countBad(const int *H) {
  long Bad = 0;
  for (int I = 0; I < N; ++I)
    Bad += H[I] != I + 1;
  return Bad;
}

// Runs Body on the main thread while a helper thread has launched slowFill(P)
// on its hipStreamPerThread and not synchronized.
template <class F> static void withHelper(int *P, F Body) {
  std::atomic<bool> Launched{false}, Release{false};
  std::thread A([&] {
    slowFill<<<1, 1, 0, hipStreamPerThread>>>(P);
    Launched = true;
    while (!Release)
      std::this_thread::yield();
  });
  while (!Launched)
    std::this_thread::yield();
  Body();
  Release = true;
  A.join();
}

static bool legacyStream() {
  int *H, *D;
  hipHostMalloc((void **)&H, sizeof(int) * N);
  hipMalloc((void **)&D, sizeof(int) * N);
  for (int I = 0; I < N; ++I)
    H[I] = -1;
  hipMemset(D, 0, sizeof(int) * N);
  hipDeviceSynchronize();
  withHelper(D, [&] {
    hipMemcpyAsync(H, D, sizeof(int) * N, hipMemcpyDeviceToHost, nullptr);
    hipStreamSynchronize(nullptr);
  });
  bool Ok = countBad(H) == 0;
  hipHostFree(H);
  hipFree(D);
  return Ok;
}

static bool deviceSync() {
  int *H;
  hipHostMalloc((void **)&H, sizeof(int) * N);
  for (int I = 0; I < N; ++I)
    H[I] = -1;
  bool Ok = false;
  withHelper(H, [&] {
    hipDeviceSynchronize();
    Ok = countBad(H) == 0;
  });
  hipHostFree(H);
  return Ok;
}

int main() {
  bool Legacy = legacyStream();
  bool Dev = deviceSync();
  printf("legacy stream: %s\n", Legacy ? "ok" : "bad");
  printf("hipDeviceSynchronize: %s\n", Dev ? "ok" : "bad");
  puts(Legacy && Dev ? "PASS" : "FAIL");
  return Legacy && Dev ? 0 : 1;
}
