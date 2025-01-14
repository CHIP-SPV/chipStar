#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(1); \
        } \
    } while(0)

// Kernel to test __builtin_amdgcn_ds_bpermute
__global__ void test_bpermute(int* output) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % 32;  // Get lane ID within wavefront
    
    // Each thread will have its thread ID as its value
    int my_value = tid;
    
    // We'll try to get the value from the next lane (rotating back to 0 at the end)
    int src_lane = ((lane_id + 1) % 32) * 4;  // Multiply by 4 since it's byte offset
    
    // Perform the bpermute operation
    int result = __builtin_amdgcn_ds_bpermute(src_lane, my_value);
    
    // Store the result
    output[tid] = result;
}

int main() {
    const int num_threads = 64;  // Two wavefronts
    std::vector<int> h_output(num_threads);
    
    // Allocate device memory
    int* d_output;
    HIP_CHECK(hipMalloc(&d_output, num_threads * sizeof(int)));
    
    // Launch kernel
    hipLaunchKernelGGL(test_bpermute, 
                       dim3(1), 
                       dim3(num_threads), 
                       0, 0, 
                       d_output);
    
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy results back
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, num_threads * sizeof(int), hipMemcpyDeviceToHost));
    
    // Verify results
    std::cout << "Results:\n";
    for (int i = 0; i < num_threads; i++) {
        int expected = (i % 32 == 31) ? (i - 31) : (i + 1);  // Wrap around within each wavefront
        std::cout << "Thread " << i << ": got " << h_output[i] 
                  << ", expected " << expected 
                  << (h_output[i] == expected ? " ✓" : " ✗") << "\n";
    }
    
    // Cleanup
    HIP_CHECK(hipFree(d_output));
    return 0;
}