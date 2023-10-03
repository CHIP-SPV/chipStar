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

__global__ void checkValues(int* data, int* result) {
       int tid = threadIdx.x;
       int value = data[tid];

       // Check if any value in the thread warp is greater than 5
       int any_greater_than_5 = (__any(value > 5)) ? 1 : 0;

       if (tid == 0) {
               *result = any_greater_than_5;
       }

}

int main() {
       int num_elements = 8;
       int data[] = {1, 2, 3, 4, 5, 6, 7, 8};
       int result;
       int* d_data;
       int* d_result;

       // Allocate and copy data to the device
       cudaMalloc((void**)&d_data, sizeof(int) * num_elements);
       cudaMalloc((void**)&d_result, sizeof(int));
       cudaMemcpy(d_data, data, sizeof(int) * num_elements, cudaMemcpyHostToDevice);

       // Launch the kernel
       checkValues<<<1, num_elements>>>(d_data, d_result);
       cudaDeviceSynchronize();

       // Copy the result back to the host
       cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

       if (result) {
               printf("At least one value in the thread warp is greater than 5\n");
       } else {
               printf("No value in the thread warp is greater than 5\n");
       }

       // Cleanup
       cudaFree(d_data);
       cudaFree(d_result);

       return 0;
}
