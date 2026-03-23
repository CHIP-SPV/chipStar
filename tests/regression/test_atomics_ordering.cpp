// Test cross-workgroup atomic visibility.
// With memory_order_relaxed, this may produce wrong results on some
// platforms due to lack of cross-workgroup ordering guarantees.
// With memory_order_seq_cst, the final count must be exactly N*blocks.

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void atomic_count(unsigned int *counter) {
    atomicAdd(counter, 1u);
}

int main() {
    const int NBLOCKS = 256;
    const int NTHREADS = 256;
    unsigned int *d_counter;
    (void)hipMalloc(&d_counter, sizeof(unsigned int));
    (void)hipMemset(d_counter, 0, sizeof(unsigned int));

    atomic_count<<<NBLOCKS, NTHREADS>>>(d_counter);
    (void)hipDeviceSynchronize();

    unsigned int result;
    (void)hipMemcpy(&result, d_counter, sizeof(unsigned int), hipMemcpyDeviceToHost);

    unsigned int expected = NBLOCKS * NTHREADS;
    int pass = (result == expected);
    printf("atomicAdd count: %u (expected %u) => %s\n", result, expected, pass ? "PASS" : "FAIL");
    (void)hipFree(d_counter);
    return pass ? 0 : 1;
}
