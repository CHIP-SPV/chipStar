// Reproduces #1990: OpenCL reports maxGridSize 65536 although it runs larger
// grids.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

__global__ void mark(int *Out) {
  Out[blockIdx.x + blockIdx.y + blockIdx.z] = 1;
}

int main() {
  constexpr unsigned N = 70000;
  hipDeviceProp_t Props;
  int *Out = nullptr;
  if (hipGetDeviceProperties(&Props, 0) != hipSuccess ||
      hipMalloc(&Out, N * sizeof(int)) != hipSuccess) {
    std::printf("FAIL: setup\n");
    return 1;
  }

  for (int D = 0; D < 3; ++D) {
    unsigned G[3] = {1, 1, 1};
    G[D] = N;
    std::vector<int> Host(N);
    if (hipMemset(Out, 0, N * sizeof(int)) != hipSuccess) {
      std::printf("FAIL: memset\n");
      return 1;
    }
    hipLaunchKernelGGL(mark, dim3(G[0], G[1], G[2]), dim3(1), 0, 0, Out);
    if (hipMemcpy(Host.data(), Out, N * sizeof(int),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      std::printf("FAIL: dim %d launch or copy\n", D);
      return 1;
    }
    for (unsigned I = 0; I < N; ++I)
      if (Host[I] != 1) {
        std::printf("FAIL: dim %d block %u did not run\n", D, I);
        return 1;
      }
    if (Props.maxGridSize[D] < static_cast<int>(N)) {
      std::printf("FAIL: maxGridSize[%d] = %d, but %u blocks ran\n", D,
                  Props.maxGridSize[D], N);
      return 1;
    }
  }
  std::printf("PASS\n");
  return 0;
}
