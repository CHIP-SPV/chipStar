#include <hip/hip_runtime.h>

__global__ void kernel() {
    printf("Hello World!");
}