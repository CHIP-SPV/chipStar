// Times constant-offset __shfl, __shfl_down and __shfl_up against the same
// loops with the lane or delta loaded from memory (CHIP-SPV/chipStar#1978).

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t E = (call);                                                     \
    if (E != hipSuccess) {                                                     \
      printf("FAIL: %s at %s:%d\n", hipGetErrorString(E), __FILE__, __LINE__); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void rotate(int *Out, const int *Lane, int Iters) {
  int V = threadIdx.x * 7;
  for (int I = 0; I < Iters; I++)
    V = __shfl(V, threadIdx.x + 1) + 1;
  Out[blockIdx.x * blockDim.x + threadIdx.x] = V;
}

__global__ void rotateOpaque(int *Out, const int *Lane, int Iters) {
  int L = Lane[blockIdx.x * blockDim.x + threadIdx.x];
  int V = threadIdx.x * 7;
  for (int I = 0; I < Iters; I++)
    V = __shfl(V, L) + 1;
  Out[blockIdx.x * blockDim.x + threadIdx.x] = V;
}

__global__ void down(int *Out, const int *Delta, int Iters) {
  int V = threadIdx.x * 7;
  for (int I = 0; I < Iters; I++)
#pragma unroll
    for (int D = 16; D > 0; D /= 2)
      V = __shfl_down(V, D) + 1;
  Out[blockIdx.x * blockDim.x + threadIdx.x] = V;
}

__global__ void downOpaque(int *Out, const int *Delta, int Iters) {
  int D0 = Delta[0], D1 = Delta[1], D2 = Delta[2], D3 = Delta[3],
      D4 = Delta[4];
  int V = threadIdx.x * 7;
  for (int I = 0; I < Iters; I++) {
    V = __shfl_down(V, D0) + 1;
    V = __shfl_down(V, D1) + 1;
    V = __shfl_down(V, D2) + 1;
    V = __shfl_down(V, D3) + 1;
    V = __shfl_down(V, D4) + 1;
  }
  Out[blockIdx.x * blockDim.x + threadIdx.x] = V;
}

// A jacobi-style warp sum.
__global__ void sum(int *Out, const int *Delta, int Iters) {
  float V = threadIdx.x;
  for (int I = 0; I < Iters; I++)
#pragma unroll
    for (int D = 8; D > 0; D /= 2)
      V = 0.5f * (V + __shfl_down(V, D));
  Out[blockIdx.x * blockDim.x + threadIdx.x] = __float_as_int(V);
}

__global__ void sumOpaque(int *Out, const int *Delta, int Iters) {
  int D0 = Delta[0], D1 = Delta[1], D2 = Delta[2], D3 = Delta[3];
  float V = threadIdx.x;
  for (int I = 0; I < Iters; I++) {
    V = 0.5f * (V + __shfl_down(V, D0));
    V = 0.5f * (V + __shfl_down(V, D1));
    V = 0.5f * (V + __shfl_down(V, D2));
    V = 0.5f * (V + __shfl_down(V, D3));
  }
  Out[blockIdx.x * blockDim.x + threadIdx.x] = __float_as_int(V);
}

// An nbnxm-style transposing reduction of three values with up and down.
__device__ float transpose(float F0, float F1, float F2, int D1, int D2,
                           int D4) {
  F0 = 0.5f * (F0 + __shfl_down(F0, D1));
  F1 = 0.5f * (F1 + __shfl_up(F1, D1));
  F2 = 0.5f * (F2 + __shfl_down(F2, D1));
  if (threadIdx.x & 1)
    F0 = F1;
  F0 = 0.5f * (F0 + __shfl_down(F0, D2));
  F2 = 0.5f * (F2 + __shfl_up(F2, D2));
  if (threadIdx.x & 2)
    F0 = F2;
  return 0.5f * (F0 + __shfl_down(F0, D4));
}

template <bool Opaque>
__global__ void mixed(int *Out, const int *Delta, int Iters) {
  int D1 = Opaque ? Delta[0] : 1, D2 = Opaque ? Delta[1] : 2,
      D4 = Opaque ? Delta[2] : 4;
  float F = threadIdx.x;
  for (int I = 0; I < Iters; I++)
    F = transpose(F, F + 1, F + 2, D1, D2, D4);
  Out[blockIdx.x * blockDim.x + threadIdx.x] = __float_as_int(F);
}

using Kernel = decltype(&rotate);
static const int Block = 256;
static int Grid, N;

// Host wall clock: some drivers' event timestamps are unusable (UHD 770 OpenCL).
static float timeMs(Kernel K, int *Out, const int *In, int Iters) {
  auto T0 = std::chrono::steady_clock::now();
  K<<<Grid, Block>>>(Out, In, Iters);
  CHECK(hipDeviceSynchronize());
  auto T1 = std::chrono::steady_clock::now();
  return std::chrono::duration<float, std::milli>(T1 - T0).count();
}

// Returns the speedup of Fast over Opaque, or 0 if their results differ.
static float measure(const char *What, Kernel Fast, Kernel Opaque,
                     const std::vector<int> &HIn) {
  int *Out, *In;
  CHECK(hipMalloc(&Out, 2 * N * sizeof(int)));
  CHECK(hipMalloc(&In, HIn.size() * sizeof(int)));
  CHECK(hipMemcpy(In, HIn.data(), HIn.size() * sizeof(int),
                  hipMemcpyHostToDevice));
  // Grow the work until the opaque loop runs for 50 ms, far above timer noise;
  // 2^24 iterations of at most 5 increments stay below INT_MAX.
  int Iters = 16;
  timeMs(Opaque, Out, In, Iters);
  timeMs(Fast, Out + N, In, Iters);
  while (Iters < (1 << 24) && timeMs(Opaque, Out, In, Iters) < 50.0f)
    Iters *= 2;
  const int Reps = 7;
  std::vector<float> TF, TO;
  for (int R = 0; R < Reps; R++) {
    TO.push_back(timeMs(Opaque, Out, In, Iters));
    TF.push_back(timeMs(Fast, Out + N, In, Iters));
  }
  std::sort(TF.begin(), TF.end());
  std::sort(TO.begin(), TO.end());
  std::vector<int> H(2 * N);
  CHECK(hipMemcpy(H.data(), Out, 2 * N * sizeof(int), hipMemcpyDeviceToHost));
  CHECK(hipFree(Out));
  CHECK(hipFree(In));
  printf("%s: %d threads x %d iterations: constant %.3f ms, opaque %.3f ms, "
         "speedup %.2fx\n",
         What, N, Iters, TF[Reps / 2], TO[Reps / 2], TO[Reps / 2] / TF[Reps / 2]);
  if (memcmp(H.data(), H.data() + N, N * sizeof(int))) {
    printf("FAIL: %s: constant and opaque results differ\n", What);
    return 0;
  }
  return TO[Reps / 2] / TF[Reps / 2];
}

int main() {
  hipDeviceProp_t P;
  CHECK(hipGetDeviceProperties(&P, 0));
  Grid = 4 * P.multiProcessorCount;
  N = Grid * Block;
  std::vector<int> Lane(N);
  for (int I = 0; I < N; I++)
    Lane[I] = I % Block + 1;
  float Rot = measure("rotate", rotate, rotateOpaque, Lane);
  float Red = measure("down", down, downOpaque, {16, 8, 4, 2, 1});
  float Sum = measure("sum", sum, sumOpaque, {8, 4, 2, 1});
  float Mix = measure("mixed", mixed<false>, mixed<true>, {1, 2, 4});

  const char *Env = getenv("CHIP_FAST_SHUFFLE");
  bool Off = Env && (!strcasecmp(Env, "off") || !strcmp(Env, "0") ||
                     !strcasecmp(Env, "false"));
  // Battlemage and PVC gather an indexed shuffle through a0; other devices
  // only report.
  bool Gathers = strstr(P.name, "Arc(TM) B") || strstr(P.name, "GPU Max");
  printf("%s, fast shuffle %s\n", P.name, Off ? "off" : "on");
  if (!Rot || !Red || !Sum || !Mix)
    return 1;
  float Min = std::min({Rot, Red, Sum, Mix});
  if (!Off && Gathers && Min < 1.5f) {
    printf("FAIL: on this GPU the constant forms must be 1.5x faster\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}
