// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
// #include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <stdio.h>

__global__ void countBits(int* data) {
       int tid = threadIdx.x;
       int value = data[tid];

       // Count the number of set bits in the thread warp
       int bit_count = __popc(__ballot(value));

       if (tid == 0) {
               printf("Number of set bits in the thread warp: %d\n", bit_count);
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
       countBits<<<1, num_elements>>>(d_data);
       cudaDeviceSynchronize();

       // Cleanup
       cudaFree(d_data);

       return 0;
}
