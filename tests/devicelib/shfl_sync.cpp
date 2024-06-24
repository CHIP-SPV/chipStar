#include <hip/hip_runtime.h>
#include <iostream>

__global__ void test_shfl_sync(int *data, int *result, unsigned mask) {
    int laneId = threadIdx.x % warpSize;

    int value = data[threadIdx.x];

    int shfl_val = __shfl_sync(mask, value, 0);
    int shfl_up_val = __shfl_up_sync(mask, value, 1);
    int shfl_down_val = __shfl_down_sync(mask, value, 1);
    int shfl_xor_val = __shfl_xor_sync(mask, value, 1);

    if (laneId == 0) {
        result[0] = shfl_val;
        result[1] = shfl_up_val;
        result[2] = shfl_down_val;
        result[3] = shfl_xor_val;
    }
}

void check_results(int *result, int expected[4]) {
    bool passed = true;
    for (int i = 0; i < 4; ++i) {
        if (result[i] != expected[i]) {
            passed = false;
            break;
        }
    }
    if (passed) {
        std::cout << "PASSED!" << std::endl;
    } else {
        std::cout << "FAILED!" << std::endl;
    }
}

int main() {
    const int warpSize = 32;
    int data[warpSize];
    int result[4];

    for (int i = 0; i < warpSize; ++i) {
        data[i] = i;
    }

    int *d_data, *d_result;
    hipMalloc(&d_data, warpSize * sizeof(int));
    hipMalloc(&d_result, 4 * sizeof(int));

    hipMemcpy(d_data, data, warpSize * sizeof(int), hipMemcpyHostToDevice);

    unsigned masks[3] = {0x00000000, 0xFFFFFFFF, 0x55555555};
    // The mask 0x55555555 makes every other thread participate because:
    // In binary, it's 01010101010101010101010101010101
    // Each '1' bit corresponds to an active thread, and each '0' bit to an inactive thread.
    // This pattern repeats every two bits, effectively enabling every other thread in the warp.
    int expected_results[3][4] = {
        {0, 0, 0, 0}, // Expected results for mask 0x00000000
        {0, 0, 1, 1}, // Expected results for mask 0xFFFFFFFF
        {1, 0, 1, 0}  // Expected results for mask 0x55555555
    };

    for (int i = 0; i < 3; ++i) {
        hipLaunchKernelGGL(test_shfl_sync, dim3(1), dim3(warpSize), 0, 0, d_data, d_result, masks[i]);
        hipMemcpy(result, d_result, 4 * sizeof(int), hipMemcpyDeviceToHost);
        hipDeviceSynchronize();

        std::cout << "Results for mask " << std::hex << masks[i] << std::dec << ":" << std::endl;
        std::cout << "shfl_sync: " << result[0] << std::endl;
        std::cout << "shfl_up_sync: " << result[1] << std::endl;
        std::cout << "shfl_down_sync: " << result[2] << std::endl;
        std::cout << "shfl_xor_sync: " << result[3] << std::endl;

        check_results(result, expected_results[i]);
    }

    hipFree(d_data);
    hipFree(d_result);

    return 0;
}
