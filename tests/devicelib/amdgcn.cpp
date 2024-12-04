#include <hip/hip_runtime.h>
#include <cassert>
#include <stdio.h>

#define WARP_SIZE 32
#define CHECK_HIP(cmd) \
  do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
      printf("HIP error: '%s'(%d) at %s:%d\n", \
             hipGetErrorString(error), error, __FILE__, __LINE__); \
      return EXIT_FAILURE; \
    } \
  } while(0)

// Test kernel for wave barrier
__global__ void testWaveBarrierKernel() {
  __builtin_amdgcn_wave_barrier();
}

// Test kernel for memory fence 
__global__ void testFenceKernel(int* data) {
  data[threadIdx.x] = threadIdx.x;
  __builtin_amdgcn_fence(1, ""); // System scope fence
  data[threadIdx.x] += 1;
}

// Test kernel for ds_bpermute
__global__ void testBPermuteKernel(int* output) {
  int myVal = threadIdx.x;
  // Rotate values left by 1 position within wave
  int rotated = __builtin_amdgcn_ds_bpermute(1, myVal);
  output[threadIdx.x] = rotated;
}

int main() {
  // Test wave barrier
  testWaveBarrierKernel<<<1,WARP_SIZE>>>();
  CHECK_HIP(hipDeviceSynchronize());

  // Test memory fence
  int *d_data;
  int h_data[WARP_SIZE];
  CHECK_HIP(hipMalloc(&d_data, WARP_SIZE * sizeof(int)));
  testFenceKernel<<<1,WARP_SIZE>>>(d_data);
  CHECK_HIP(hipMemcpy(h_data, d_data, WARP_SIZE * sizeof(int), hipMemcpyDeviceToHost));
  
  for (int i = 0; i < WARP_SIZE; i++) {
    if (h_data[i] != i + 1) {
      printf("Memory fence test failed at index %d: expected %d, got %d\n", 
             i, i + 1, h_data[i]);
      return EXIT_FAILURE;
    }
  }

  // Test ds_bpermute
  int *d_output;
  int h_output[WARP_SIZE];
  CHECK_HIP(hipMalloc(&d_output, WARP_SIZE * sizeof(int)));
  testBPermuteKernel<<<1,WARP_SIZE>>>(d_output);
  CHECK_HIP(hipMemcpy(h_output, d_output, WARP_SIZE * sizeof(int), hipMemcpyDeviceToHost));
  
  // Each thread should get value from thread to its left
  // Thread 0 gets value from last thread in the warp
  for (int i = 0; i < WARP_SIZE; i++) {
    int expected = (i + WARP_SIZE - 1) % WARP_SIZE;
    if (h_output[i] != expected) {
      printf("Permute test failed at index %d: expected %d, got %d\n", 
             i, expected, h_output[i]);
      return EXIT_FAILURE;
    }
  }

  CHECK_HIP(hipFree(d_data));
  CHECK_HIP(hipFree(d_output));
  return EXIT_SUCCESS;
}
