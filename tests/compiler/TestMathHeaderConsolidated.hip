/**
 * @file TestMathHeaderConsolidated.hip
 * @brief Reproducer for various math function compilation issues
 * 
 * This test reproduces multiple issues:
 * 1. rocSPARSE ceil function compilation issue where rocSPARSE tries to call std::ceilf
 *    from __host__ __device__ code, but only __device__ versions of ceilf are available.
 * 2. min/max function conflicts between __host__ __device__ and __device__ declarations
 * 
 * The errors we're reproducing:
 * - "error: reference to __device__ function 'ceilf' in __host__ __device__ function"
 * - "error: __host__ __device__ function 'min' cannot overload __device__ function 'min'"
 */

#include <hip/hip_runtime.h>
#include <cmath>

// Test case 1: Reproduce the min/max overload conflict
// This reproduces the error when both __device__ and __host__ __device__ versions exist
namespace test_minmax_conflict {

// These would conflict with existing __device__ declarations in int_math.hh
// extern __device__ int min(int a, int b);  // Already declared in int_math.hh
// __host__ __device__ inline static int min(int arg1, int arg2) { ... }  // Causes conflict

__host__ __device__ void test_minmax_in_host_device_function() {
    int a = 5, b = 10;
    // This should work if min/max are properly declared as __host__ __device__
    int min_result = min(a, b);
    int max_result = max(a, b);
    
    // Prevent unused variable warnings
    (void)min_result;
    (void)max_result;
}

} // namespace test_minmax_conflict

// Test case 2: Reproduce rocSPARSE's namespace and function pattern exactly
namespace rocsparse {

// This mirrors rocSPARSE's common.h pattern - calling std::ceilf from __host__ __device__
__device__ __host__ __forceinline__ float ceil(float x) { 
    return std::ceilf(x); 
}

__device__ __host__ __forceinline__ double ceil(double x) { 
    return std::ceil(x); 
}

__device__ __host__ __forceinline__ float floor(float x) { 
    return std::floorf(x); 
}

__device__ __host__ __forceinline__ double floor(double x) { 
    return std::floor(x); 
}

} // namespace rocsparse

// This function reproduces the rocSPARSE host code pattern that causes the error
__host__ __device__ void reproduce_rocsparse_error() {
    // This line reproduces the exact call pattern from rocSPARSE line 483:
    // = rocsparse::ceil((float)info->lrb.nRowsBins[j] / block_size);
    
    float row_length = 1000.0f;
    const int BLOCK_SIZE = 256;
    
    // This should fail compilation if std::ceilf resolves to a device-only function
    // because it would try to call a __device__ function from __host__ __device__ code
    float result_ceil = rocsparse::ceil(row_length / BLOCK_SIZE);
    
    // Test floor functions as well
    float result_floor = rocsparse::floor(row_length / BLOCK_SIZE);
    
    // Test double versions
    double result_ceil_d = rocsparse::ceil((double)row_length / BLOCK_SIZE);
    double result_floor_d = rocsparse::floor((double)row_length / BLOCK_SIZE);
    
    // Prevent unused variable warnings
    (void)result_ceil;
    (void)result_floor;
    (void)result_ceil_d;
    (void)result_floor_d;
}

// Simple device kernel to test device-side compilation
__global__ void test_device_math(float* input, float* output_ceil, float* output_floor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // This should work fine in device code
        output_ceil[idx] = rocsparse::ceil(input[idx]);
        output_floor[idx] = rocsparse::floor(input[idx]);
        
        // Test min/max in device code too
        int a = 5, b = 10;
        int min_result = min(a, b);
        int max_result = max(a, b);
        (void)min_result;
        (void)max_result;
    }
}

int main() {
    // Test 1: Reproduce the host-side compilation issue
    reproduce_rocsparse_error();
    
    // Test 2: Test min/max in host device function
    test_minmax_conflict::test_minmax_in_host_device_function();
    
    // Test 3: Verify device-side compilation works  
    const int n = 1;
    float *d_input, *d_output_ceil, *d_output_floor;
    hipMalloc(&d_input, n * sizeof(float));
    hipMalloc(&d_output_ceil, n * sizeof(float));
    hipMalloc(&d_output_floor, n * sizeof(float));
    
    float input_val = 3.7f;
    hipMemcpy(d_input, &input_val, sizeof(float), hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(test_device_math, dim3(1), dim3(1), 0, 0, d_input, d_output_ceil, d_output_floor, n);
    hipDeviceSynchronize();
    
    float result_ceil, result_floor;
    hipMemcpy(&result_ceil, d_output_ceil, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(&result_floor, d_output_floor, sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_input);
    hipFree(d_output_ceil);
    hipFree(d_output_floor);
    
    // Check if we got the expected ceiling and floor values
    bool success = (result_ceil == 4.0f && result_floor == 3.0f);
    
    return success ? 0 : 1;
} 