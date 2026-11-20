// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2410: a float4
// access to a shared array of narrower elements must not need an i128.
#include <hip/hip_runtime.h>

#include <cstdio>

constexpr int N = 64;

// One float per lane: a wide load of scalar stores, then scalar loads of a
// wide store.
__global__ void floatLanes(float4 *Loaded, float *Stored) {
  __shared__ alignas(float4) float Buf[4 * N];
  int T = threadIdx.x;
  for (int K = 0; K < 4; ++K)
    Buf[4 * T + K] = 4 * T + K;
  __syncthreads();
  float4 V = reinterpret_cast<float4 *>(Buf)[(T + 1) % N];
  Loaded[T] = V * make_float4(1, 2, 3, 4);
  __syncthreads();
  reinterpret_cast<float4 *>(Buf)[T] =
      make_float4(1000 + T, 2000 + T, 3000 + T, 4000 + T);
  __syncthreads();
  for (int K = 0; K < 4; ++K)
    Stored[4 * T + K] = Buf[4 * T + K];
}

// Two lanes per 64-bit element, the same way.
__global__ void wordLanes(uint4 *Loaded, unsigned long long *Stored) {
  __shared__ alignas(uint4) unsigned long long Buf[2 * N];
  int T = threadIdx.x;
  for (int K = 0; K < 2; ++K)
    Buf[2 * T + K] = ((4ull * T + 2 * K + 1) << 32) | (4 * T + 2 * K);
  __syncthreads();
  uint4 V = reinterpret_cast<uint4 *>(Buf)[(T + 1) % N];
  Loaded[T] = V * make_uint4(1, 2, 3, 4);
  __syncthreads();
  reinterpret_cast<uint4 *>(Buf)[T] =
      make_uint4(1000 + T, 2000 + T, 3000 + T, 4000 + T);
  __syncthreads();
  for (int K = 0; K < 2; ++K)
    Stored[2 * T + K] = Buf[2 * T + K];
}

int main() {
  float4 *FL;
  float *FS;
  uint4 *UL;
  unsigned long long *US;
  if (hipMalloc(&FL, N * sizeof(float4)) != hipSuccess ||
      hipMalloc(&FS, 4 * N * sizeof(float)) != hipSuccess ||
      hipMalloc(&UL, N * sizeof(uint4)) != hipSuccess ||
      hipMalloc(&US, 2 * N * sizeof(unsigned long long)) != hipSuccess)
    return 1;
  floatLanes<<<1, N>>>(FL, FS);
  wordLanes<<<1, N>>>(UL, US);
  float4 HFL[N];
  float HFS[4 * N];
  uint4 HUL[N];
  unsigned long long HUS[2 * N];
  if (hipMemcpy(HFL, FL, sizeof(HFL), hipMemcpyDeviceToHost) != hipSuccess ||
      hipMemcpy(HFS, FS, sizeof(HFS), hipMemcpyDeviceToHost) != hipSuccess ||
      hipMemcpy(HUL, UL, sizeof(HUL), hipMemcpyDeviceToHost) != hipSuccess ||
      hipMemcpy(HUS, US, sizeof(HUS), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  int Errors = 0;
  auto Check = [&](bool Ok, const char *What, int T) {
    if (!Ok && Errors++ < 8)
      printf("thread %d: wrong %s\n", T, What);
  };
  for (int T = 0; T < N; ++T) {
    unsigned B = 4 * ((T + 1) % N);
    Check(HFL[T].x == B && HFL[T].y == 2 * (B + 1) &&
              HFL[T].z == 3 * (B + 2) && HFL[T].w == 4 * (B + 3),
          "float4 load", T);
    Check(HUL[T].x == B && HUL[T].y == 2 * (B + 1) &&
              HUL[T].z == 3 * (B + 2) && HUL[T].w == 4 * (B + 3),
          "uint4 load", T);
    Check(HFS[4 * T] == 1000 + T && HFS[4 * T + 1] == 2000 + T &&
              HFS[4 * T + 2] == 3000 + T && HFS[4 * T + 3] == 4000 + T,
          "float4 store", T);
    Check(HUS[2 * T] == ((2000ull + T) << 32 | (1000 + T)) &&
              HUS[2 * T + 1] == ((4000ull + T) << 32 | (3000 + T)),
          "uint4 store", T);
  }
  hipFree(FL);
  hipFree(FS);
  hipFree(UL);
  hipFree(US);
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}
