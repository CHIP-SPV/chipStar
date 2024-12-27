#include <hip/hip_runtime.h>
#include <cassert>
#include <stdio.h>

#define WARP_SIZE 32
#define GLOBAL_SIZE 64
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
  if (threadIdx.x < WARP_SIZE) {
    int myVal = threadIdx.x;
    int rotated = __builtin_amdgcn_ds_bpermute(1, myVal);
    output[threadIdx.x] = rotated;
  }
}

// Test kernel for ballot and population count
__global__ void testBallotPopcKernel(int* output) {
  if (threadIdx.x < WARP_SIZE) {
    unsigned long long ballot_result = __ballot(threadIdx.x % 2 == 0);
    
    if (threadIdx.x == 0) {
      output[0] = ballot_result;
    }
    
    if (threadIdx.x == 1) {
      output[1] = __popc((unsigned int)ballot_result);
      output[2] = __popcll(ballot_result);
    }
  }
}

// Test kernel for ffs (find first set bit)
__global__ void testFfsKernel(int* output) {
  if (threadIdx.x == 0) {
    // Test various patterns
    output[0] = __ffs(0);          // Should return 0 (no bits set)
    output[1] = __ffs(1);          // Should return 1 (bit 0 set)
    output[2] = __ffs(0x80000000); // Should return 32 (bit 31 set)
    output[3] = __ffs(0x100);      // Should return 9 (bit 8 set)
  }
}

// Test kernel for shuffle operations
__global__ void testShuffleKernel(int* output) {
  if (threadIdx.x < WARP_SIZE) {
    int myVal = threadIdx.x;
    
    int broadcast_val = __shfl(myVal, 0);
    output[threadIdx.x] = broadcast_val;
    
    int shift_val = __shfl_up(myVal, 1);
    output[threadIdx.x + WARP_SIZE] = shift_val;
    
    int shift_down_val = __shfl_down(myVal, 1);
    output[threadIdx.x + 2*WARP_SIZE] = shift_down_val;
  }
}

int main() {
  // Test wave barrier
  testWaveBarrierKernel<<<1,GLOBAL_SIZE>>>();
  CHECK_HIP(hipDeviceSynchronize());

  // Test memory fence
  int *d_data;
  int h_data[GLOBAL_SIZE];
  CHECK_HIP(hipMalloc(&d_data, GLOBAL_SIZE * sizeof(int)));
  testFenceKernel<<<1,GLOBAL_SIZE>>>(d_data);
  CHECK_HIP(hipMemcpy(h_data, d_data, GLOBAL_SIZE * sizeof(int), hipMemcpyDeviceToHost));
  
  for (int i = 0; i < GLOBAL_SIZE; i++) {
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
  testBPermuteKernel<<<1,GLOBAL_SIZE>>>(d_output);
  CHECK_HIP(hipMemcpy(h_output, d_output, WARP_SIZE * sizeof(int), hipMemcpyDeviceToHost));
  
  // Each thread should get value from thread to its left
  // Thread 0 gets value from last thread in the warp
  for (int i = 0; i < WARP_SIZE; i++) {
    int expected = (i + 1) % WARP_SIZE;
    if (h_output[i] != expected) {
      printf("Permute test failed at index %d: expected %d, got %d\n", 
             i, expected, h_output[i]);
      return EXIT_FAILURE;
    }
  }

  // Test ballot and population count
  int *d_ballot_output;
  int h_ballot_output[3];  // Need space for ballot result and two popc results
  CHECK_HIP(hipMalloc(&d_ballot_output, 3 * sizeof(int)));
  
  testBallotPopcKernel<<<1,GLOBAL_SIZE>>>(d_ballot_output);
  CHECK_HIP(hipMemcpy(h_ballot_output, d_ballot_output, 3 * sizeof(int), hipMemcpyDeviceToHost));

  // Verify ballot result - should have bits set for even thread IDs
  unsigned long long expected_ballot = 0;
  for (int i = 0; i < WARP_SIZE; i++) {
    if (i % 2 == 0) {
      expected_ballot |= (1ULL << i);
    }
  }
  
  if ((unsigned long long)h_ballot_output[0] != expected_ballot) {
    printf("Ballot test failed: expected %llx, got %llx\n", 
           expected_ballot, (unsigned long long)h_ballot_output[0]);
    return EXIT_FAILURE;
  }

  // Verify popc results
  int expected_popc = WARP_SIZE / 2;  // Half of threads are even
  if (h_ballot_output[1] != expected_popc) {
    printf("POPC test failed: expected %d, got %d\n", 
           expected_popc, h_ballot_output[1]);
    return EXIT_FAILURE;
  }
  
  if (h_ballot_output[2] != expected_popc) {
    printf("POPCLL test failed: expected %d, got %d\n", 
           expected_popc, h_ballot_output[2]);
    return EXIT_FAILURE;
  }

  // Test ffs
  int *d_ffs_output;
  int h_ffs_output[4];
  CHECK_HIP(hipMalloc(&d_ffs_output, 4 * sizeof(int)));
  
  testFfsKernel<<<1,GLOBAL_SIZE>>>(d_ffs_output);
  CHECK_HIP(hipMemcpy(h_ffs_output, d_ffs_output, 4 * sizeof(int), hipMemcpyDeviceToHost));

  // Verify ffs results
  if (h_ffs_output[0] != 0) {
    printf("FFS test failed for 0: expected 0, got %d\n", h_ffs_output[0]);
    return EXIT_FAILURE;
  }
  if (h_ffs_output[1] != 1) {
    printf("FFS test failed for 1: expected 1, got %d\n", h_ffs_output[1]);
    return EXIT_FAILURE;
  }
  if (h_ffs_output[2] != 32) {
    printf("FFS test failed for 0x80000000: expected 32, got %d\n", h_ffs_output[2]);
    return EXIT_FAILURE;
  }
  if (h_ffs_output[3] != 9) {
    printf("FFS test failed for 0x100: expected 9, got %d\n", h_ffs_output[3]);
    return EXIT_FAILURE;
  }

  // Test shuffle operations
  int *d_shfl_output;
  int h_shfl_output[WARP_SIZE * 3];  // Space for 3 tests
  CHECK_HIP(hipMalloc(&d_shfl_output, 3 * WARP_SIZE * sizeof(int)));
  
  testShuffleKernel<<<1,GLOBAL_SIZE>>>(d_shfl_output);
  CHECK_HIP(hipMemcpy(h_shfl_output, d_shfl_output, 3 * WARP_SIZE * sizeof(int), hipMemcpyDeviceToHost));

  // Verify broadcast results (all threads should have value 0)
  for (int i = 0; i < WARP_SIZE; i++) {
    if (h_shfl_output[i] != 0) {
      printf("Shuffle broadcast test failed at index %d: expected 0, got %d\n",
             i, h_shfl_output[i]);
      return EXIT_FAILURE;
    }
  }

  // Verify shift up results (first thread should have undefined value, others get i-1)
  for (int i = 1; i < WARP_SIZE; i++) {
    if (h_shfl_output[i + WARP_SIZE] != i-1) {
      printf("Shuffle up test failed at index %d: expected %d, got %d\n",
             i, i-1, h_shfl_output[i + WARP_SIZE]);
      return EXIT_FAILURE;
    }
  }

  // Verify shift down results (last thread should have undefined value, others get i+1)
  for (int i = 0; i < WARP_SIZE-1; i++) {
    if (h_shfl_output[i + 2*WARP_SIZE] != i+1) {
      printf("Shuffle down test failed at index %d: expected %d, got %d\n",
             i, i+1, h_shfl_output[i + 2*WARP_SIZE]);
      return EXIT_FAILURE;
    }
  }

  CHECK_HIP(hipFree(d_data));
  CHECK_HIP(hipFree(d_output));
  CHECK_HIP(hipFree(d_ballot_output));
  CHECK_HIP(hipFree(d_ffs_output));
  CHECK_HIP(hipFree(d_shfl_output));
  return EXIT_SUCCESS;
}
