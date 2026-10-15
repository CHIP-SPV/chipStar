// __shfl from a constant lane offset (a lane rotate), and __shfl_down and
// __shfl_up by a constant delta, must give the same result as the same
// shuffle with the lane or delta loaded from memory, for every offset, width,
// element type and block shape (CHIP-SPV/chipStar#1978). Blocks whose
// threadIdx.x does not follow the sub-group lane take the generic rotate.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t E = (call);                                                     \
    if (E != hipSuccess) {                                                     \
      printf("FAIL: %s at %s:%d\n", hipGetErrorString(E), __FILE__, __LINE__); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

constexpr int Iters = 5;

__device__ int linearTid() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

template <typename T, int C, int W>
__global__ void rotate(T *Out, const int *Lane) {
  int Tid = linearTid();
  T V = (T)(Tid * 3 + 1);
  for (int I = 0; I < Iters; I++)
    V = (W ? __shfl(V, threadIdx.x + C, W) : __shfl(V, threadIdx.x + C)) +
        (T)1;
  Out[Tid] = V;
}

template <typename T, int C, int W>
__global__ void opaque(T *Out, const int *Lane) {
  int Tid = linearTid();
  int L = Lane[Tid];
  T V = (T)(Tid * 3 + 1);
  for (int I = 0; I < Iters; I++)
    V = (W ? __shfl(V, L, W) : __shfl(V, L)) + (T)1;
  Out[Tid] = V;
}

// Every delta 1..31 in turn, unrolled so each one is a constant.
template <typename T, int W, bool Up, bool Opaque>
__global__ void shift(T *Out, const int *Delta) {
  int Tid = linearTid();
  T V = (T)(Tid * 3 + 1), S = 0;
#pragma unroll
  for (int D = 1; D < 32; D++) {
    unsigned E = Opaque ? Delta[D] : D;
    V = (Up ? (W ? __shfl_up(V, E, W) : __shfl_up(V, E))
            : (W ? __shfl_down(V, E, W) : __shfl_down(V, E))) +
        (T)D;
    S += V;
  }
  Out[Tid] = S;
}

static int WarpSize, Bad;

// Lanes to compare: a partial sub-group's lanes read lanes that do not exist,
// unless the sub-group holds whole segments of width W.
template <int W> int compared(int N) {
  return W && N % WarpSize % W == 0 ? N : N - N % WarpSize;
}

template <typename T, int C, int W> void check(dim3 B) {
  if (W > WarpSize)
    return;
  int N = B.x * B.y * B.z;
  std::vector<int> HLane(N);
  for (int I = 0; I < N; I++)
    HLane[I] = I % B.x + C;
  T *Out;
  int *Lane;
  CHECK(hipMalloc(&Out, 2 * N * sizeof(T)));
  CHECK(hipMalloc(&Lane, N * sizeof(int)));
  CHECK(hipMemcpy(Lane, HLane.data(), N * sizeof(int), hipMemcpyHostToDevice));
  rotate<T, C, W><<<1, B>>>(Out, Lane);
  opaque<T, C, W><<<1, B>>>(Out + N, Lane);
  CHECK(hipGetLastError());
  std::vector<T> H(2 * N);
  CHECK(hipMemcpy(H.data(), Out, 2 * N * sizeof(T), hipMemcpyDeviceToHost));
  CHECK(hipFree(Out));
  CHECK(hipFree(Lane));
  for (int I = 0; I < compared<W>(N); I++)
    if (memcmp(&H[I], &H[N + I], sizeof(T))) {
      printf("FAIL: C=%d W=%d block %ux%ux%u thread %d: rotate %g, opaque %g\n",
             C, W, B.x, B.y, B.z, I, (double)H[I], (double)H[N + I]);
      Bad++;
      return;
    }
}

template <typename T, int W, bool Up> void checkShift(dim3 B) {
  if (W > WarpSize)
    return;
  int N = B.x * B.y * B.z;
  std::vector<int> HDelta(32);
  for (int D = 0; D < 32; D++)
    HDelta[D] = D;
  T *Out;
  int *Delta;
  CHECK(hipMalloc(&Out, 2 * N * sizeof(T)));
  CHECK(hipMalloc(&Delta, 32 * sizeof(int)));
  CHECK(hipMemcpy(Delta, HDelta.data(), 32 * sizeof(int),
                  hipMemcpyHostToDevice));
  shift<T, W, Up, false><<<1, B>>>(Out, Delta);
  shift<T, W, Up, true><<<1, B>>>(Out + N, Delta);
  CHECK(hipGetLastError());
  std::vector<T> H(2 * N);
  CHECK(hipMemcpy(H.data(), Out, 2 * N * sizeof(T), hipMemcpyDeviceToHost));
  CHECK(hipFree(Out));
  CHECK(hipFree(Delta));
  for (int I = 0; I < compared<W>(N); I++)
    if (memcmp(&H[I], &H[N + I], sizeof(T))) {
      printf("FAIL: %s W=%d block %ux%ux%u thread %d: constant %g, opaque %g\n",
             Up ? "up" : "down", W, B.x, B.y, B.z, I, (double)H[I],
             (double)H[N + I]);
      Bad++;
      return;
    }
}

// Every launch shape costs a JIT compile per kernel on PoCL, so the full set
// runs for int only.
template <typename T> void checkType(bool Full) {
  for (dim3 B : {dim3(100), dim3(16, 16)}) {
    check<T, 1, 0>(B);
    if (Full) {
      check<T, 3, 0>(B);
      check<T, 31, 0>(B);
      check<T, 33, 0>(B);
      check<T, 3, 16>(B);
      check<T, 5, 8>(B);
    }
  }
  checkShift<T, 0, false>(dim3(100));
  checkShift<T, 0, true>(dim3(100));
  if (!Full)
    return;
  checkShift<T, 16, false>(dim3(100));
  checkShift<T, 16, true>(dim3(100));
  checkShift<T, 8, false>(dim3(100));
  checkShift<T, 8, true>(dim3(100));
  check<T, 1, 4>(dim3(4));
  checkShift<T, 4, false>(dim3(4));
  checkShift<T, 4, true>(dim3(4));
}

int main() {
  hipDeviceProp_t P;
  CHECK(hipGetDeviceProperties(&P, 0));
  WarpSize = P.warpSize;
  checkType<int>(true);
  checkType<unsigned>(false);
  checkType<long>(false);
  checkType<float>(false);
  if (!Bad)
    printf("PASSED\n");
  return Bad ? 1 : 0;
}
