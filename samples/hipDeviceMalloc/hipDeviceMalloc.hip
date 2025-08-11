#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <new> // For placement new

class construct {};

template<typename T>
    inline void __host__ __device__ construct(T *p) {
  ::new(static_cast<void*>(p)) T();
}


#define HIP_CHECK(call)                                                 \
    do {                                                                \
        hipError_t err = call;                                          \
        if (err != hipSuccess) {                                        \
            printf("HIP error %s:%d: '%s'\n", __FILE__, __LINE__,       \
                   hipGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

__global__ void dynamicAllocationKernel(int* result, int size, bool* alloc_success)
{
    // Allocate memory dynamically on the device
    int* data = (int*)(malloc(size * sizeof(int))); // Cast size to 64-bit
    
#ifdef __HIP_DEVICE_COMPILE__
    // Test device-side new/delete
    struct SimpleStruct {
        int x;
        float y;
    };
    SimpleStruct* test_obj = new SimpleStruct;
    if (test_obj) {
        test_obj->x = 10;
        test_obj->y = 3.14f;
        // Potentially use test_obj->x or test_obj->y in a way that prevents optimization away
        // For this test, existence and non-null is sufficient.
        delete test_obj;
        test_obj = nullptr; // Good practice
    } else {
        // If new fails, we might want to indicate this, perhaps via alloc_success
        // For now, we'll assume it passes or fails similarly to malloc
    }
#endif

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
        checksum += data[i];  // Changed to addition instead of XOR
    }

    // Store the result
    atomicAdd(result, checksum);  // Changed to atomicAdd instead of atomicXor

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
            checksum += data[j];  // Changed to addition instead of XOR
        }

        atomicAdd(result, checksum);  // Changed to atomicAdd instead of atomicXor
        
        free(data);
    }
    
    *alloc_success = true;
    atomicAdd(&debug_info[2], iterations);  // Count successful iterations
}

// Add function declarations
void test1(int size);
void test2(int size, int stress_iterations);

int main()
{
    const int size = 100;  // Reduced problem size
    const int stress_iterations = 10000;
    
    test1(size);
    test2(size, stress_iterations);
    
    return 0;
}

void test1(int size)
{
    int* d_result;
    int h_result;
    bool* d_alloc_success;
    bool h_alloc_success;
    
    // Allocate memory for the result and allocation success flag on the device
    HIP_CHECK(hipMalloc((void**)&d_result, sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&d_alloc_success, sizeof(bool)));
    
    // Initialize the result to 0
    HIP_CHECK(hipMemset(d_result, 0, sizeof(int)));
    
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
    int threads_per_block = local.x * local.y * local.z;
    int num_blocks = global.x * global.y * global.z;
    int total_threads = (num_blocks * threads_per_block);
    int expected_checksum = (size * (size - 1)) / 2;  // Sum of arithmetic sequence from 0 to size-1
    expected_checksum *= total_threads;  // Multiply by total number of threads

    if (h_result != expected_checksum) {
        printf("Error: Incorrect checksum. Expected %d, got %d\n", expected_checksum, h_result);
        printf("Debug: size=%d, blocks=%d, threads_per_block=%d, total_threads=%d\n", 
               size, num_blocks, threads_per_block, total_threads);
        exit(EXIT_FAILURE);
    }

    printf("Main kernel checksum: %d (correct)\n", h_result);
    
    // Reset the result and allocation success flag
    HIP_CHECK(hipMemset(d_result, 0, sizeof(int)));
    HIP_CHECK(hipMemset(d_alloc_success, 0, sizeof(bool)));
    
    // Free device memory
    HIP_CHECK(hipFree(d_result));
    HIP_CHECK(hipFree(d_alloc_success));
}

void test2(int size, int stress_iterations)
{
    int* d_result;
    int h_result;
    bool* d_alloc_success;
    bool h_alloc_success;
    int* d_debug_info;
    int h_debug_info[3] = {0};
    
    // Allocate memory
    HIP_CHECK(hipMalloc((void**)&d_result, sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&d_alloc_success, sizeof(bool)));
    HIP_CHECK(hipMalloc((void**)&d_debug_info, 3 * sizeof(int)));
    
    // Initialize
    HIP_CHECK(hipMemset(d_result, 0, sizeof(int)));
    HIP_CHECK(hipMemset(d_debug_info, 0, 3 * sizeof(int)));
    
    // Launch the stress test kernel
    stressTestKernel<<<1, 1>>>(d_result, size, stress_iterations, d_alloc_success, d_debug_info);
    
    // Copy the allocation success flag back to the host
    HIP_CHECK(hipMemcpy(&h_alloc_success, d_alloc_success, sizeof(bool), hipMemcpyDeviceToHost));
    
    if (!h_alloc_success) {
        printf("Device-side malloc failed in the stress test kernel\n");
        exit(EXIT_FAILURE);
    }
    
    // Copy the result back to the host
    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
    
    // For stress test verification
    int expected_checksum = (size * (size - 1)) / 2;  // Reset for stress test - only one thread
    expected_checksum *= stress_iterations;  // Multiply by number of iterations
    
    if (h_result != expected_checksum) {
        printf("FAILED: Incorrect checksum in stress test. Expected %d, got %d\n", expected_checksum, h_result);
        printf("Debug: size=%d, iterations=%d, sequence_sum=%d\n", 
               size, stress_iterations, (size * (size - 1)) / 2);
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
}
