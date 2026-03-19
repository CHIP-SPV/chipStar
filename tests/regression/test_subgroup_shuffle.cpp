// Reproducer: __chip_all() always returns false
//
// __chip_ballot() returns ulong, but with 32-wide warps the max
// value is 0x00000000FFFFFFFF. The comparison ~0 promotes to
// 0xFFFFFFFFFFFFFFFF, so __chip_all() always returned false.
//
// Expected: __all(1) returns 1 when all threads pass
// Bug: __all(1) returns 0

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void test_all_kernel(int *result) {
    int tid = threadIdx.x;
    // All threads pass predicate=1 → __all should return 1
    result[0] = __all(1);
    // One thread passes 0 → __all should return 0
    result[1] = __all(tid != 0 ? 1 : 0);
}

int main() {
    int *d, h[2];
    (void)hipMalloc(&d, 2 * sizeof(int));
    test_all_kernel<<<1, 32>>>(d);
    (void)hipDeviceSynchronize();
    (void)hipMemcpy(h, d, 2 * sizeof(int), hipMemcpyDeviceToHost);

    int pass = (h[0] == 1 && h[1] == 0);
    printf("__all(1)=%d (expect 1), __all(mixed)=%d (expect 0) => %s\n",
           h[0], h[1], pass ? "PASS" : "FAIL");
    (void)hipFree(d);
    return pass ? 0 : 1;
}
