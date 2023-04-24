#include <hip/hip_runtime.h>

__global__ void kernel() {
    printf("Hello World!");
}

int main() {
    kernel<<<1, 1>>>();
    hipDeviceSynchronize();
    return 0;
}
