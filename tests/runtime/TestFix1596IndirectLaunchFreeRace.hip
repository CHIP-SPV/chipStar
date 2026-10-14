// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1596: a launch
// with indirect access racing a free from a kernel completion callback.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void k(int **P) { atomicAdd(*P, 1); }

int main() {
  int *B, **P, H = 0;
  (void)hipMalloc(&B, sizeof(int));
  (void)hipMemset(B, 0, sizeof(int));
  (void)hipMalloc(&P, sizeof(int *));
  (void)hipMemcpy(P, &B, sizeof(B), hipMemcpyHostToDevice);
  const int N = 2000;
  for (int I = 0; I < N; ++I) {
    int *A;
    (void)hipMalloc(&A, 1 << 16);
    k<<<1, 1>>>(P);
    (void)hipFree(A); // A now dies when the launch's keep-alives are dropped.
    k<<<1, 1>>>(P);
    hipError_t E = hipGetLastError();
    if (E != hipSuccess) {
      std::printf("FAIL: iteration %d: %s\n", I, hipGetErrorString(E));
      return 1;
    }
  }
  (void)hipMemcpy(&H, B, sizeof(H), hipMemcpyDeviceToHost);
  bool Ok = H == 2 * N;
  std::printf("%s: %d of %d\n", Ok ? "PASS" : "FAIL", H, 2 * N);
  return !Ok;
}
