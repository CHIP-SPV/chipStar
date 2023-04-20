#include <hip/hip_runtime.h>

__global__ void kernel() {
    printf(HIP_VERSION_MAJOR);
    printf(HIP_VERSION_MINOR);
    printf(HIP_VERSION_PATCH);
}

int main() {
    kernel<<<1, 1>>>();
    hipDeviceSynchronize();
    return 0;
}

