#include <iostream>
#include <hip/hip_runtime.h>

// Simple time-consuming kernel without arguments
__global__ void slowKernel() {
    float val = 0.0f;
    for (int i = 0; i < 100000000; i++) {
        for (int j = 0; j < 10000; j++) {
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
    assert(hipEventElapsedTime(&milliseconds, start, start) == hipSuccess);
    return 0;
}