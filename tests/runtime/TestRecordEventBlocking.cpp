#include <iostream>
#include <chrono>
#include <hip/hip_runtime.h>

// Simple time-consuming kernel without arguments
__global__ void slowKernel() {
  float val = 0.0f;
  for (int i = 0; i < 10000; i++) {
    for (int j = 0; j < 100000; j++) {
      val += sqrtf(val + i + j);
    }
  }
}

int main() {
  float milliseconds = 0;
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventRecord(start, 0);
  // hipEventElapsedTime(&milliseconds, start, start);
  hipLaunchKernelGGL(slowKernel, dim3(512), dim3(256), 0, 0);
  // time how long it takes to execute to check if blocking or not
  auto startTime = std::chrono::high_resolution_clock::now();
  hipEventElapsedTime(&milliseconds, start, start);
  auto endTime = std::chrono::high_resolution_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime)
          .count();
  if (ms > 1000) {
    std::cout << "FAIL: Blocking" << std::endl;
    exit(1);
  }
  else {
    std::cout << "PASS: Non-Blocking" << std::endl;
    exit(0);
  }

  return 0;
}
