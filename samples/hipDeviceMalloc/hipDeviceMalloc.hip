#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define DEVICE_HEAP_SIZE 1024 * 1024 // 1 MB

#define HIP_CHECK(call)                                                 \
    do {                                                                \
        hipError_t err = call;                                          \
        if (err != hipSuccess) {                                        \
            printf("HIP error %s:%d: '%s'\n", __FILE__, __LINE__,       \
                   hipGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// __global__ void init_device_heap_kernel(void* device_heap) {
//     __chip_init_device_heap(device_heap);
// }

__global__ void dynamicAllocationKernel(int* result, int size, bool* alloc_success)
{
    // Allocate memory dynamically on the device
    int* data = (int*)(malloc(size * sizeof(int))); // Cast size to 64-bit
    
    if (data == NULL) {
        *alloc_success = false;
        return;
    }
    
    *alloc_success = true;
    
    // Initialize the allocated memory
    for (int i = 0; i < size; i++) {
        data[i] = i;
    }
    
    // Compute a checksum of the array
    int checksum = 0;
    for (int i = 0; i < size; i++) {
        checksum ^= data[i];  // XOR operation to compute checksum
    }

    // Store the result
    atomicXor(result, checksum);  // Use atomicXor for checksum

    // Free the dynamically allocated memory
    free(data);
}

__global__ void stressTestKernel(int* result, int size, int iterations, bool* alloc_success, int* debug_info)
{
    for (int i = 0; i < iterations; i++) {
        int* data = (int*)malloc(size * sizeof(int));
        if (data == NULL) {
            printf("Device-side malloc failed in stress test, iteration %d\n", i);
            *alloc_success = false;
            atomicAdd(&debug_info[0], 1);  // Count failed allocations
            atomicAdd(&debug_info[1], i);  // Store the last failed iteration
            return;
        }
        
        for (int j = 0; j < size; j++) {
            data[j] = j;
        }
        
        int checksum = 0;
        for (int j = 0; j < size; j++) {
            checksum ^= data[j];  // XOR operation to compute checksum
        }

        atomicXor(result, checksum);  // Use atomicXor for checksum
        
        free(data);
    }
    
    *alloc_success = true;
    atomicAdd(&debug_info[2], iterations);  // Count successful iterations
}

int main()
{
    const int size = 100;  // Reduced problem size
    const int stress_iterations = 10000;
    int* d_result;
    int h_result;
    bool* d_alloc_success;
    bool h_alloc_success;
    int* d_debug_info;
    int h_debug_info[3] = {0};  // [0]: failed allocations, [1]: last failed iteration, [2]: successful iterations
    
    void* device_heap;
    HIP_CHECK(hipMalloc(&device_heap, DEVICE_HEAP_SIZE));

    // init_device_heap_kernel<<<1, 1>>>(device_heap);
    // HIP_CHECK(hipDeviceSynchronize());

    // Allocate memory for the result and allocation success flag on the device
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_alloc_success, sizeof(bool)));
    HIP_CHECK(hipMalloc(&d_debug_info, 3 * sizeof(int)));
    
    // Initialize the result to 0
    HIP_CHECK(hipMemset(d_result, 0, sizeof(int)));
    HIP_CHECK(hipMemset(d_debug_info, 0, 3 * sizeof(int)));
    
    // Launch the kernel
    dim3 global(4, 4, 4);
    dim3 local(2, 2, 2);
    dynamicAllocationKernel<<<global, local>>>(d_result, size, d_alloc_success);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy the allocation success flag back to the host
    HIP_CHECK(hipMemcpy(&h_alloc_success, d_alloc_success, sizeof(bool), hipMemcpyDeviceToHost));
    
    if (!h_alloc_success) {
        printf("Device-side malloc failed in the main kernel\n");
        exit(EXIT_FAILURE);
    }
    
    // Copy the result back to the host
    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
    
    // Check correctness using checksum
    int num_blocks = (global.x * global.y * global.z) / (local.x * local.y * local.z);
    int expected_checksum = 0;
    for (int i = 0; i < size; i++) {
        expected_checksum ^= i;  // Compute expected checksum
    }
    expected_checksum *= num_blocks;

    if (h_result != expected_checksum) {
        printf("Error: Incorrect checksum. Expected %d, got %d\n", expected_checksum, h_result);
        exit(EXIT_FAILURE);
    }

    printf("Main kernel checksum: %d (correct)\n", h_result);
    
    // Reset the result and allocation success flag
    HIP_CHECK(hipMemset(d_result, 0, sizeof(int)));
    HIP_CHECK(hipMemset(d_alloc_success, 0, sizeof(bool)));
    
    // Launch the stress test kernel
    stressTestKernel<<<1, 1>>>(d_result, size, stress_iterations, d_alloc_success, d_debug_info);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy the allocation success flag back to the host
    HIP_CHECK(hipMemcpy(&h_alloc_success, d_alloc_success, sizeof(bool), hipMemcpyDeviceToHost));
    
    if (!h_alloc_success) {
        printf("Device-side malloc failed in the stress test kernel\n");
        exit(EXIT_FAILURE);
    }
    
    // Copy the result back to the host
    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
    
    // Check correctness for stress test using checksum
    expected_checksum *= stress_iterations;
    if (h_result != expected_checksum) {
        printf("FAILED: Incorrect checksum in stress test. Expected %d, got %d\n", expected_checksum, h_result);
        exit(EXIT_FAILURE);
    }

    printf("Stress test checksum: %d (correct)\n", h_result);
    printf("Stress test completed successfully\n");
    
    // Copy debug info back to host
    HIP_CHECK(hipMemcpy(h_debug_info, d_debug_info, 3 * sizeof(int), hipMemcpyDeviceToHost));
    
    printf("Debug Info:\n");
    printf("  Failed allocations: %d\n", h_debug_info[0]);
    printf("  Last failed iteration: %d\n", h_debug_info[1]);
    printf("  Successful iterations: %d\n", h_debug_info[2]);
    printf("PASSED\n");
    
    // Free device memory
    HIP_CHECK(hipFree(d_result));
    HIP_CHECK(hipFree(d_alloc_success));
    HIP_CHECK(hipFree(d_debug_info));
    
    return 0;
}
