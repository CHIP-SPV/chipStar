// Pointers chosen at run time: read through a pointer-valued __device__
// variable, an atomic through a pointer to shared or global memory, and a
// table of buffer pointers indexed by thread.
#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %s (line %d)\n", #Call, hipGetErrorName(Err),  \
             __LINE__);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)
__device__ float Buf[64];
__device__ float *Gp = Buf;
__global__ void fill() { Buf[threadIdx.x] = threadIdx.x * 2.0f; }
__global__ void viaGlobal(float *o) { o[threadIdx.x] = Gp[threadIdx.x]; }
__global__ void mixedAtomic(int *g, int c) {
  __shared__ int s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  int *p = c ? &s : g;
  atomicAdd(p, 1);
  __syncthreads();
  if (threadIdx.x == 0 && c) g[1] = s;
}
__global__ void table(const float *a, const float *b, float *o) {
  const float *src[2] = {a, b};
  int i = threadIdx.x;
  o[i] = src[i & 1][i];
}
int main() {
  int bad = 0;
  float *o, h[64];
  CHECK(hipMalloc(&o, sizeof h));
  fill<<<1, 64>>>(); CHECK(hipGetLastError()); viaGlobal<<<1, 64>>>(o); CHECK(hipGetLastError());
  CHECK(hipMemcpy(h, o, sizeof h, hipMemcpyDeviceToHost));
  for (int i = 0; i < 64; ++i) if (h[i] != i * 2.0f && bad++ < 3) printf("global %d: %g\n", i, h[i]);
  int *g, hg[2];
  CHECK(hipMalloc(&g, 8)); CHECK(hipMemset(g, 0, 8));
  mixedAtomic<<<1, 64>>>(g, 0); CHECK(hipGetLastError()); mixedAtomic<<<1, 64>>>(g, 1); CHECK(hipGetLastError());
  CHECK(hipMemcpy(hg, g, 8, hipMemcpyDeviceToHost));
  if (hg[0] != 64 || hg[1] != 64) { printf("atomic: %d %d\n", hg[0], hg[1]); ++bad; }
  float *a, *b, ha[64], hb[64];
  for (int i = 0; i < 64; ++i) { ha[i] = i; hb[i] = 1000 + i; }
  CHECK(hipMalloc(&a, sizeof ha)); CHECK(hipMalloc(&b, sizeof hb));
  CHECK(hipMemcpy(a, ha, sizeof ha, hipMemcpyHostToDevice)); CHECK(hipMemcpy(b, hb, sizeof hb, hipMemcpyHostToDevice));
  table<<<1, 64>>>(a, b, o); CHECK(hipGetLastError());
  CHECK(hipMemcpy(h, o, sizeof h, hipMemcpyDeviceToHost));
  for (int i = 0; i < 64; ++i) if (h[i] != (i & 1 ? hb[i] : ha[i]) && bad++ < 6) printf("table %d: %g\n", i, h[i]);
  printf("%s\n", bad ? "FAILED" : "PASSED");
  return bad != 0;
}
