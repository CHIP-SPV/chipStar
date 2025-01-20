#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void testWarpCalc(int* debug) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int globalIdx = bid * blockDim.x + tid;
    
    // Do some computation to prevent optimization
    int result = 0;
    for(int i = 0; i < tid + 1; i++) {
        result += i * globalIdx;
    }
    
    // Store using atomic operation
    atomicExch(&debug[globalIdx], result);
}

int main() {
    const int gridSize = 4;
    const int blockSize = 64;
    const int numThreads = gridSize * blockSize;

    // Allocate pinned memory
    int* h_debug;
    hipHostMalloc(&h_debug, numThreads * sizeof(int));
    memset(h_debug, 0, numThreads * sizeof(int));

    // Allocate device memory
    int* d_debug;
    hipMalloc(&d_debug, numThreads * sizeof(int));
    hipMemset(d_debug, 0, numThreads * sizeof(int));

    dim3 grid(gridSize);
    dim3 block(blockSize);
    
    printf("Launching kernel with grid=%d, block=%d\n\n", gridSize, blockSize);
    
    // Use triple angle bracket syntax
    testWarpCalc<<<grid, block>>>(d_debug);
    hipDeviceSynchronize();
    
    // Copy results back
    hipMemcpy(h_debug, d_debug, numThreads * sizeof(int), hipMemcpyDeviceToHost);
    
    printf("Results for first few threads:\n");
    printf("GlobalIdx\tValue\n");
    
    // Print first few entries
    for (int i = 0; i < 8; i++) {
        printf("%d\t\t%d\n", i, h_debug[i]);
    }

    // Cleanup
    hipHostFree(h_debug);
    hipFree(d_debug);
    
    return 0;
} 