// Reproduces: @llvm.used address space mismatch between HIP module (ptr addrspace(4))
// and chipStar device library (ptr addrspace(0)) causes "Appending variables with
// different element types" link error.
//
// The __attribute__((used)) __device__ static variable creates an @llvm.used entry
// with ptr addrspace(4) in the HIP compilation. The device library (hipspv-spirv64.bc)
// has @llvm.used entries with ptr (addrspace 0) from its OpenCL C compilation.
// LLVM's linker refuses to merge them.

#include <hip/hip_runtime.h>
#include <cstdio>

// This triggers @llvm.used with ptr addrspace(4) — same pattern as
// rocThrust's runtime_static_assert.h
__attribute__((used)) __device__ static int* device_ptr = nullptr;

__global__ void kernel(int* out) {
    out[threadIdx.x] = threadIdx.x;
}

int main() {
    int *d, h[32];
    hipMalloc(&d, 128);
    kernel<<<1, 32>>>(d);
    hipDeviceSynchronize();
    hipMemcpy(h, d, 128, hipMemcpyDeviceToHost);
    hipFree(d);

    bool pass = true;
    for (int i = 0; i < 32; i++) {
        if (h[i] != i) { pass = false; break; }
    }
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
