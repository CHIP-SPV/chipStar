#include <hip/hip_runtime.h>

extern __global__ void kernel();

int main() {
    kernel<<<1, 1>>>();
    hipDeviceSynchronize();
    return 0;
}
