// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
// #include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


__global__ void shiftValues(int* data) {
    int tid = threadIdx.x;
    int value = data[tid];

    // Perform an upward shift of values within the thread warp
    int shifted_value = __shfl_up(value, 1);

    if (tid != 0) {
        printf("Thread %d received value %d from Thread %d\n", tid, shifted_value, tid - 1);
    } else {
        printf("Thread 0 retains its original value %d\n", value);
    }
}

int main() {
    int num_elements = 8;
    int data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int* d_data;

    // Allocate and copy data to the device
    cudaMalloc((void**)&d_data, sizeof(int) * num_elements);
    cudaMemcpy(d_data, data, sizeof(int) * num_elements, cudaMemcpyHostToDevice);

    // Launch the kernel
    shiftValues<<<1, num_elements>>>(d_data);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_data);

    return 0;
}

