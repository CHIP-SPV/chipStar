// __all and __any vote over the lanes a warp really has: a block of half a
// warp, and the partial last warp of a block of one and a half warps.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %s (line %d)\n", #Call, hipGetErrorName(Err),  \
             __LINE__);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void votes(int *Out) {
  int T = threadIdx.x;
  Out[T] = __all(1) | (__any(T == 5) << 1) | (__all(T != 3) << 2);
}

int main() {
  hipDeviceProp_t Props;
  CHECK(hipGetDeviceProperties(&Props, 0));
  int W = Props.warpSize, Bad = 0;
  for (int N : {W / 2, W + W / 2}) {
    int *D;
    std::vector<int> H(N);
    CHECK(hipMalloc(&D, N * sizeof(int)));
    votes<<<1, N>>>(D);
    CHECK(hipGetLastError());
    CHECK(hipMemcpy(H.data(), D, N * sizeof(int), hipMemcpyDeviceToHost));
    for (int T = 0; T < N; ++T) {
      int Lo = T / W * W, Hi = Lo + W < N ? Lo + W : N;
      bool Has5 = Lo <= 5 && 5 < Hi, Has3 = Lo <= 3 && 3 < Hi;
      int Exp = 1 | (Has5 << 1) | (!Has3 << 2);
      if (H[T] != Exp && Bad++ < 4)
        printf("block %d lane %d: %d, expected %d\n", N, T, H[T], Exp);
    }
    CHECK(hipFree(D));
  }
  printf("%s\n", Bad ? "FAILED" : "PASSED");
  return Bad != 0;
}
