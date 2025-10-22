// Reproducer for Issue #977: hipMemcpyToSymbol with templated __constant__ memory
#include <hip/hip_runtime.h>
#include <cstdio>

#define MAX_MASK_WIDTH 10

template <typename T>
__constant__ T mask[MAX_MASK_WIDTH];

template <typename T>
__global__ void readMaskKernel(T* output) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < MAX_MASK_WIDTH) {
    output[tid] = mask<T>[tid];
  }
}

template <typename T>
static int runTestOnce() {
  T hostMask[MAX_MASK_WIDTH] = {};
  // Initialize with known values
  for (int i = 0; i < MAX_MASK_WIDTH; i++) {
    hostMask[i] = static_cast<T>(i + 1);
  }
  
  hipError_t st = hipMemcpyToSymbol(mask<T>, hostMask, 3 * sizeof(T));
  if (st != hipSuccess) {
    std::printf("FAIL: hipMemcpyToSymbol error=%d\n", static_cast<int>(st));
    return 1;
  }
  
  // Launch kernel to read back the values
  T* d_output;
  st = hipMalloc(&d_output, MAX_MASK_WIDTH * sizeof(T));
  if (st != hipSuccess) {
    std::printf("FAIL: hipMalloc error=%d\n", static_cast<int>(st));
    return 1;
  }
  
  hipLaunchKernelGGL(readMaskKernel<T>, dim3(1), dim3(MAX_MASK_WIDTH), 0, 0, d_output);
  
  st = hipDeviceSynchronize();
  if (st != hipSuccess) {
    std::printf("FAIL: hipDeviceSynchronize error=%d\n", static_cast<int>(st));
    hipFree(d_output);
    return 1;
  }
  
  // Copy result back and verify
  T result[MAX_MASK_WIDTH] = {};
  st = hipMemcpy(result, d_output, MAX_MASK_WIDTH * sizeof(T), hipMemcpyDeviceToHost);
  if (st != hipSuccess) {
    std::printf("FAIL: hipMemcpy error=%d\n", static_cast<int>(st));
    hipFree(d_output);
    return 1;
  }
  
  // Verify first 3 elements (what we copied)
  for (int i = 0; i < 3; i++) {
    if (result[i] != static_cast<T>(i + 1)) {
      std::printf("FAIL: Expected %f, got %f at index %d\n", 
                  static_cast<double>(static_cast<T>(i + 1)), 
                  static_cast<double>(result[i]), i);
      hipFree(d_output);
      return 1;
    }
  }
  
  hipFree(d_output);
  std::printf("PASS\n");
  return 0;
}

int main() {
  // Minimal device presence check
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) {
    std::puts("SKIP: No HIP device available");
    return 77; // CHIP_SKIP_TEST
  }

  // The issue was reported with double; use double to maximize chances to repro
  return runTestOnce<double>();
}


