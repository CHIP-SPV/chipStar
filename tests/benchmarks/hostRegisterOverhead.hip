#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
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

static constexpr int N_KERNELS = 40000;
static constexpr size_t BUF_SIZE = 1024 * 1024 * 4; // 1 MB, enough for a realistic double buffer

#ifdef INDIRECT_KERNEL
__global__ void bench_kernel(double **ptr) { (*ptr)[0] += 1.0; }
#define KERNEL_LAUNCH(ptr) hipLaunchKernelGGL(bench_kernel, dim3(1), dim3(1), 0, 0, ptr)
#define KERNEL_LABEL "indirect-ptr kernels"
#else
__global__ void bench_kernel() {}
#define KERNEL_LAUNCH(ptr) hipLaunchKernelGGL(bench_kernel, dim3(1), dim3(1), 0, 0)
#define KERNEL_LABEL "empty kernels      "
#endif

int main() {
  CHECK(hipSetDevice(0));
  
  std::vector<double> timings_registered(N_KERNELS);
  std::vector<double> timings_unregistered(N_KERNELS);
  
  double *HostBuf = (double *)malloc(BUF_SIZE);
  memset(HostBuf, 0, BUF_SIZE);

  double *DevBuf;
  double **DevPtrPtr;
  CHECK(hipMalloc(&DevBuf, sizeof(double)));
  CHECK(hipMalloc(&DevPtrPtr, sizeof(double *)));
  CHECK(hipMemcpy(DevPtrPtr, &DevBuf, sizeof(double *), hipMemcpyHostToDevice));

  // --- with hipHostRegister ---
  CHECK(hipHostRegister(HostBuf, BUF_SIZE, 0));

  CHECK(hipDeviceSynchronize());
  for (int i = 0; i < N_KERNELS; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    KERNEL_LAUNCH(DevPtrPtr);
    CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    timings_registered[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  CHECK(hipHostUnregister(HostBuf));

  // --- without hipHostRegister ---
  CHECK(hipDeviceSynchronize());
  for (int i = 0; i < N_KERNELS; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    KERNEL_LAUNCH(DevPtrPtr);
    CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    timings_unregistered[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  free(HostBuf);
  CHECK(hipFree(DevBuf));
  CHECK(hipFree(DevPtrPtr));

  double with_ms    = std::accumulate(timings_registered.begin(),timings_registered.end(),0.0);
  double without_ms = std::accumulate(timings_unregistered.begin(),timings_unregistered.end(),0.0);

  double min_with_ms    = *std::min_element(timings_registered.begin(),timings_registered.end());
  double min_without_ms = *std::min_element(timings_unregistered.begin(),timings_unregistered.end());
  double max_with_ms    = *std::max_element(timings_registered.begin(),timings_registered.end());
  double max_without_ms = *std::max_element(timings_unregistered.begin(),timings_unregistered.end());

  std::cout << N_KERNELS << " " KERNEL_LABEL " WITH    hipHostRegister: "
            << with_ms << " ms  (avg,min,max: " << with_ms / N_KERNELS
	    << ", " << min_with_ms << ", " << max_with_ms << " ms)\n";
  std::cout << N_KERNELS << " " KERNEL_LABEL " WITHOUT hipHostRegister: "
            << without_ms << " ms  (avg,min,max: " << without_ms / N_KERNELS
	    << ", " << min_without_ms << ", " << max_without_ms << " ms)\n";
  std::cout << "overhead factor (avg): " << with_ms / without_ms << "x\n";

  if(  with_ms / without_ms > 5 ) {
      printf( "Fail");
      return 1;
  }
  else {
      printf( "Pass\n");
      return 0;
  }

}
