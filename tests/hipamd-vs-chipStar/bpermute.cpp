#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>

__global__ void test_bpermute(int* output) {
    const unsigned int threadIdx_x = threadIdx.x;
    const unsigned int warp_id = threadIdx_x / 32;
    const unsigned int lane_id = threadIdx_x % 32;
    
    // Test with the same values we saw in the debug output
    int word;
    if (lane_id < 16) {
        word = 1068827891;  // Value we saw in the original debug output
    } else {
        word = 1057803469;  // Another value we saw
    }
    
    // Test different src_lane values as seen in the debug
    int src_lane;
    if (lane_id < 16) {
        src_lane = 33 + lane_id;  // Test lanes 33-48
    } else {
        src_lane = 49 + (lane_id - 16);  // Test lanes 49-64
    }
    
    // printf("Pre-bpermute: lane %d, word %d = %d, thread %d, warp %d\n", 
    //        src_lane, 0, word, threadIdx_x, warp_id);
    
    // Perform the bpermute operation
    int result = __builtin_amdgcn_ds_bpermute(src_lane << 2, word);
    
    // printf("Post-bpermute: lane %d, word %d = %d, thread %d, warp %d\n", 
    //        src_lane, 0, result, threadIdx_x, warp_id);
    
    // Store results
    output[threadIdx_x] = result;
}

int main() {
    const int num_threads = 64;
    int *d_output, *h_output;
    
    // Allocate memory
    h_output = new int[num_threads];
    hipMalloc(&d_output, num_threads * sizeof(int));
    
    // Launch kernel
    hipLaunchKernelGGL(test_bpermute, 
                       dim3(1), 
                       dim3(num_threads), 
                       0, 0, 
                       d_output);
    
    // Wait for kernel to finish
    hipDeviceSynchronize();
    
    // Copy results back
    hipMemcpy(h_output, d_output, num_threads * sizeof(int), hipMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\nResults:\n";
    for (int i = 0; i < num_threads; i++) {
        std::cout << "Thread " << i << ": " << h_output[i] << std::endl;
    }
    
    // Cleanup
    delete[] h_output;
    hipFree(d_output);
    
    return 0;
} 