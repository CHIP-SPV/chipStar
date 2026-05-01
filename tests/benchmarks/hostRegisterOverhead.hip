#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include "hip/hip_runtime.h"

#define CHECK(cmd)                                                             \
  do {                                                                         \
    hipError_t e = (cmd);                                                      \
    if (e != hipSuccess) {                                                     \
      std::cerr << "HIP error " << hipGetErrorString(e) << " at " << __FILE__ \
                << ":" << __LINE__ << "\n";                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static constexpr int N_KERNELS = 100;
static constexpr size_t BUF_SIZE = 1024 * 1024; // 1 MB, enough for a realistic double buffer

__global__ void empty_kernel() {}

int main() {
  CHECK(hipSetDevice(0));

  double *HostBuf = (double *)malloc(BUF_SIZE);
  memset(HostBuf, 0, BUF_SIZE);

  // --- with hipHostRegister ---
  CHECK(hipHostRegister(HostBuf, BUF_SIZE, 0));

  CHECK(hipDeviceSynchronize());
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N_KERNELS; ++i) {
    hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, 0);
  }
  CHECK(hipDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();

  CHECK(hipHostUnregister(HostBuf));

  // --- without hipHostRegister ---
  CHECK(hipDeviceSynchronize());
  auto t2 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N_KERNELS; ++i) {
    hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, 0);
  }
  CHECK(hipDeviceSynchronize());
  auto t3 = std::chrono::high_resolution_clock::now();

  free(HostBuf);

  double with_ms    = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double without_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  std::cout << N_KERNELS << " empty kernels WITH    hipHostRegister: "
            << with_ms << " ms  (" << with_ms / N_KERNELS << " ms/kernel)\n";
  std::cout << N_KERNELS << " empty kernels WITHOUT hipHostRegister: "
            << without_ms << " ms  (" << without_ms / N_KERNELS << " ms/kernel)\n";
  std::cout << "overhead factor: " << with_ms / without_ms << "x\n";


    if(  with_ms / without_ms > 10 ) {
      printf( "Fail");
      return 1;
    }
    else {
      printf( "Pass\n");
      return 0;
    }

}
