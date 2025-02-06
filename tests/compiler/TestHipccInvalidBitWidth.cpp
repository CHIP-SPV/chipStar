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
    
    atomicExch(&debug[globalIdx], result);
}

void testWarpCalcCPU(int* result, int gridSize, int blockSize) {
    int numThreads = gridSize * blockSize;
    for(int bid = 0; bid < gridSize; bid++) {
        for(int tid = 0; tid < blockSize; tid++) {
            int globalIdx = bid * blockSize + tid;
            int cpuResult = 0;
            for(int i = 0; i < tid + 1; i++) {
                cpuResult += i * globalIdx;
            }
            result[globalIdx] = cpuResult;
        }
    }
}

int main() {
    const int gridSize = 4;
    const int blockSize = 64;
    const int numThreads = gridSize * blockSize;

    int* h_debug;
    hipHostMalloc(&h_debug, numThreads * sizeof(int));
    memset(h_debug, 0, numThreads * sizeof(int));

    int* d_debug;
    hipMalloc(&d_debug, numThreads * sizeof(int));
    hipMemset(d_debug, 0, numThreads * sizeof(int));

    int* cpu_results = (int*)malloc(numThreads * sizeof(int));
    memset(cpu_results, 0, numThreads * sizeof(int));

    dim3 grid(gridSize);
    dim3 block(blockSize);
    
    printf("Launching kernel with grid=%d, block=%d\n\n", gridSize, blockSize);
    
    testWarpCalc<<<grid, block>>>(d_debug);
    hipDeviceSynchronize();
    
    hipMemcpy(h_debug, d_debug, numThreads * sizeof(int), hipMemcpyDeviceToHost);

    testWarpCalcCPU(cpu_results, gridSize, blockSize);
    
    bool passed = true;
    for (int i = 0; i < 8; i++) {
        bool match = (h_debug[i] == cpu_results[i]);
        if (!match) passed = false;
    }

    // Check all results
    for (int i = 8; i < numThreads; i++) {
        if (h_debug[i] != cpu_results[i]) {
            passed = false;
            break;
        }
    }

    printf(passed ? "PASSED" : "FAILED");

    hipHostFree(h_debug);
    hipFree(d_debug);
    free(cpu_results);
    
    return passed ? 0 : 1;
} 