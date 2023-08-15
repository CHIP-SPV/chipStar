#include <iostream>
#include <hip/hip_runtime.h>

// Simple time-consuming kernel without arguments
__global__ void slowKernel() {
    float val = 0.0f;
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 10000; j++) {
            val += sqrtf(val + i + j);
        }
    }
}

int main() {
    float milliseconds = 0;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start, 0);
    
    // Launching the kernel with arbitrary grid and block sizes
    hipLaunchKernelGGL(slowKernel, dim3(512), dim3(256), 0, 0);

    hipEventRecord(stop, 0);

    // assert(hipEventElapsedTime(&milliseconds, stop, stop) == hipErrorNotReady);
    // assert(hipEventElapsedTime(&milliseconds, start, start) == hipSuccess);
    // assert(hipEventElapsedTime(&milliseconds, stop, stop) == hipErrorNotReady);
     hipDeviceSynchronize(); 
    hipError_t err = hipEventElapsedTime(&milliseconds, start, stop);

    // Check if elapsed time returns hipErrorNotReady
    if (err == hipErrorNotReady) {
        std::cout << "Kernel still in progress..." << std::endl;
    } else {
        std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    }

    return 0;
}