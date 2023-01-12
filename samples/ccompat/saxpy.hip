#include <hip/hip_runtime.h>

extern "C" {
    __global__ void saxpy() { printf("PASSED\n"); }
}
