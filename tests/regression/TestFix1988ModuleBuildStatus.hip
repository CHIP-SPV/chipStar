// Reproduces #1988: a failed Level Zero module build must report its
// zeModuleCreate status. CTest sets build flags IGC rejects.
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <cstdio>
#include <cstring>
#include <vector>

__global__ void k(int *Out) { *Out = 1; }

int main() {
  int *D, N = 0;
  if (hipMalloc(&D, sizeof(int)) != hipSuccess ||
      hipGetBackendNativeHandles(0, nullptr, &N) != hipSuccess || N < 1)
    return 1;
  std::vector<uintptr_t> H(N);
  if (hipGetBackendNativeHandles(0, H.data(), nullptr) != hipSuccess)
    return 1;
  if (strcmp(reinterpret_cast<const char *>(H[0]), "level0")) {
    printf("HIP_SKIP_THIS_TEST: Level Zero only\n");
    return 0;
  }
  hipLaunchKernelGGL(k, dim3(1), dim3(1), 0, 0, D);
  hipError_t E = hipGetLastError();
  printf(E == hipErrorInvalidImage ? "PASS\n" : "FAIL: %s\n",
         hipGetErrorName(E));
  return E == hipErrorInvalidImage ? 0 : 1;
}
