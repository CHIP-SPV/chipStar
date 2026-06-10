// Regression test: atomicMax<unsigned int> must use unsigned comparison.
//
// chipStar's bitcode/devicelib.cl declared atomic_fetch_max_explicit with
// `uint *` instead of `atomic_uint *`. SPIRV-LLVM-Translator's
// containsUnsignedAtomicType() inspects the *Itanium mangling* and only
// recognizes an unsigned-atomic operation when the pointer is `_Atomic uint`
// (mangled `U7_Atomic`). Without that, the translator emits OpAtomicSMax
// instead of OpAtomicUMax — so `atomicMax<unsigned int>(...)` on values
// with the high bit set used signed comparison and produced wrong results
// (e.g. it considered 0x80000001u "less than" 0x7fffffffu).
//
// This test exercises the wrong-result case end-to-end on the device.
//
// See: bitcode/devicelib.cl  (DEF_CHIP_ATOMIC2 macros)

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void k_single(unsigned int *out, unsigned int v) {
    atomicMax(out, v);
}

__global__ void k_two_threads(unsigned int *out) {
    unsigned int v = (threadIdx.x == 0) ? 0x80000001u : 0xfffffff0u;
    atomicMax(out, v);
}

__global__ void k_many(unsigned int *out, const unsigned int *vals, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) atomicMax(out, vals[tid]);
}

#define CHK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %d at %s:%d\n", (int)e, __FILE__, __LINE__); \
    return 2; } } while(0)

int main() {
    int fails = 0;

    // T1: initial 0x7fffffff, atomicMax with 0x80000001 -> expect 0x80000001
    {
        unsigned int *d_out;
        CHK(hipMalloc(&d_out, sizeof(unsigned int)));
        unsigned int init = 0x7fffffffu;
        CHK(hipMemcpy(d_out, &init, sizeof(init), hipMemcpyHostToDevice));
        k_single<<<1, 1>>>(d_out, 0x80000001u);
        CHK(hipDeviceSynchronize());
        unsigned int r = 0;
        CHK(hipMemcpy(&r, d_out, sizeof(r), hipMemcpyDeviceToHost));
        unsigned int expected = 0x80000001u;
        int ok = (r == expected);
        printf("[T1] init=0x%08x arg=0x%08x got=0x%08x expected=0x%08x %s\n",
               init, 0x80000001u, r, expected, ok ? "PASS" : "FAIL");
        if (!ok) ++fails;
        CHK(hipFree(d_out));
    }

    // T2: two threads, both high-bit-set values
    {
        unsigned int *d_out;
        CHK(hipMalloc(&d_out, sizeof(unsigned int)));
        unsigned int init = 0u;
        CHK(hipMemcpy(d_out, &init, sizeof(init), hipMemcpyHostToDevice));
        k_two_threads<<<1, 2>>>(d_out);
        CHK(hipDeviceSynchronize());
        unsigned int r = 0;
        CHK(hipMemcpy(&r, d_out, sizeof(r), hipMemcpyDeviceToHost));
        unsigned int expected = 0xfffffff0u;
        int ok = (r == expected);
        printf("[T2] init=0x%08x got=0x%08x expected=0x%08x %s\n",
               init, r, expected, ok ? "PASS" : "FAIL");
        if (!ok) ++fails;
        CHK(hipFree(d_out));
    }

    // T3: many threads, mix of low/high values
    {
        const int N = 1024;
        unsigned int *d_vals, *d_out;
        CHK(hipMalloc(&d_vals, N * sizeof(unsigned int)));
        CHK(hipMalloc(&d_out, sizeof(unsigned int)));
        unsigned int *h_vals = new unsigned int[N];
        unsigned int expected = 0;
        for (int i = 0; i < N; ++i) {
            if (i == 500) h_vals[i] = 0xfffffffeu;
            else if (i % 3 == 0) h_vals[i] = 0x80000000u + (unsigned)i;
            else h_vals[i] = (unsigned)i * 7u;
            if (h_vals[i] > expected) expected = h_vals[i];
        }
        CHK(hipMemcpy(d_vals, h_vals, N * sizeof(unsigned int), hipMemcpyHostToDevice));
        unsigned int init = 0u;
        CHK(hipMemcpy(d_out, &init, sizeof(init), hipMemcpyHostToDevice));
        k_many<<<(N + 63) / 64, 64>>>(d_out, d_vals, N);
        CHK(hipDeviceSynchronize());
        unsigned int r = 0;
        CHK(hipMemcpy(&r, d_out, sizeof(r), hipMemcpyDeviceToHost));
        int ok = (r == expected);
        printf("[T3] got=0x%08x expected=0x%08x %s\n", r, expected,
               ok ? "PASS" : "FAIL");
        if (!ok) ++fails;
        delete[] h_vals;
        CHK(hipFree(d_vals));
        CHK(hipFree(d_out));
    }

    printf("\n%d test(s) failed\n", fails);
    return fails ? 1 : 0;
}
