#include <hip/hip_runtime.h>
#include <stdio.h>

// This kernel tests if printf calls with static %s specifiers are correctly handled.
// The HipPrintf pass should *not* split the call if the %s argument is a static string
// when converting to OpenCL printf.
__global__ void printf_kernel() {
    printf("File %s errored\n", "main.hip");
}

// Dummy main for compilation.
int main() {
    hipLaunchKernelGGL(printf_kernel, dim3(32), dim3(1), 0, 0);
    hipDeviceSynchronize();
    return 0;
} 