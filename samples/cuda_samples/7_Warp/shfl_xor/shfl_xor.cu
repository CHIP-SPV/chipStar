// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
// #include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


__global__ void xorValues(int* data) {
       int tid = threadIdx.x;
       int value = data[tid];

       // Perform a XOR-based shuffle within the thread warp
       int partner_tid = tid ^ 1;  // XOR operation to determine partner thread index
       int shuffled_value = __shfl_xor(value, partner_tid);

       printf("Thread %d shuffled value %d with Thread %d\n", tid, shuffled_value, partner_tid);
}

int main() {
    int num_elements = 8;
    int data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int* d_data;

    // Allocate and copy data to the device
    cudaMalloc((void**)&d_data, sizeof(int) * num_elements);
    cudaMemcpy(d_data, data, sizeof(int) * num_elements, cudaMemcpyHostToDevice);

    // Launch the kernel
    xorValues<<<1, num_elements>>>(d_data);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_data);

    return 0;
}

