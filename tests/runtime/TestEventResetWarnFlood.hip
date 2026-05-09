// Reproduces: CHIPEventLevel0::reset() warning flood.
// Re-recording the same hipEvent_t after a hipStreamSynchronize (without an
// intervening per-event hipEventQuery/hipEventSynchronize) leaves chipStar's
// host-side EventStatus_ stuck at RECORDING. Each subsequent hipEventRecord
// reaches the warn line in CHIPEventLevel0::reset(), producing
// (NUM_EVS * iters - NUM_EVS) warnings. The test runs with CHIP_LOGLEVEL=warn
// so the warning reaches stderr; ctest's FAIL_REGULAR_EXPRESSION catches it.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIPCK(x) do { hipError_t err=(x); if (err != hipSuccess) { \
  fprintf(stderr,"FAIL: HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(err)); \
  std::exit(1);} } while(0)

__global__ void touch(float *p, int n, int it) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = p[i] * 1.000001f + (float)(it & 0xff);
}

int main() {
  const int iters = 50;
  const int N = 1024;
  HIPCK(hipSetDevice(0));
  hipStream_t stream;
  HIPCK(hipStreamCreate(&stream));

  float *d_buf = nullptr;
  HIPCK(hipMalloc(&d_buf, N * sizeof(float)));

  constexpr int NUM_EVS = 8;
  hipEvent_t evs[NUM_EVS];
  for (int i = 0; i < NUM_EVS; ++i) HIPCK(hipEventCreate(&evs[i]));

  for (int it = 0; it < iters; ++it) {
    for (int k = 0; k < NUM_EVS; ++k) {
      hipLaunchKernelGGL(touch, dim3((N+255)/256), dim3(256), 0, stream,
                         d_buf, N, it);
      HIPCK(hipEventRecord(evs[k], stream));
    }
    HIPCK(hipStreamSynchronize(stream));
  }

  for (int i = 0; i < NUM_EVS; ++i) HIPCK(hipEventDestroy(evs[i]));
  HIPCK(hipFree(d_buf));
  HIPCK(hipStreamDestroy(stream));
  printf("PASS\n");
  return 0;
}
