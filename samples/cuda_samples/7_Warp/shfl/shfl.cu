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

__global__ void shuffleValues(int* data) {
       int tid = threadIdx.x;
       int value = data[tid];

       // Perform a shuffle operation within the thread warp
       int source_thread = (tid + 1) % blockDim.x;  // Circular pattern of source threads
       int shuffled_value = __shfl(value, source_thread);

       printf("Thread %d shuffled value %d with Thread %d\n", tid, shuffled_value, source_thread);
}

int main() {
       int num_elements = 8;
       int data[] = {1, 2, 3, 4, 5, 6, 7, 8};
       int* d_data;

       // Allocate and copy data to the device
       cudaMalloc((void**)&d_data, sizeof(int) * num_elements);
       cudaMemcpy(d_data, data, sizeof(int) * num_elements, cudaMemcpyHostToDevice);

       // Launch the kernel
       shuffleValues<<<1, num_elements>>>(d_data);
       cudaDeviceSynchronize();

       // Cleanup
       cudaFree(d_data);

       return 0;
}
