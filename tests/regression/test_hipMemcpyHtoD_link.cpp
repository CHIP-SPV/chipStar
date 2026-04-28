// Regression test for hipMemcpyHtoD / hipMemcpyHtoDAsync ABI mismatch.
//
// The HIP-7 submodule update changed the public signatures in
// hip_runtime_api.h to take `const void* src` (inside extern "C"), but
// src/CHIPBindings.cc still defines them with `void* Src`. The compiler
// treats those as two different functions:
//   - the header-declared extern "C" symbol has no implementation, so
//     any caller that includes the header gets an unresolved-reference
//     link error;
//   - the .cc-defined function ends up C++-mangled
//     (e.g. _Z13hipMemcpyHtoDPvS_m) and is unreachable from C and from
//     any caller that uses the public header.
//
// This test simply calls both APIs through the public header. If the
// signatures match, it links and runs cleanly (the runtime call is
// expected to fail with a non-success error since we pass null pointers,
// but that is fine — we only care that the linker resolves the symbol).
// If the bug is present, the link itself fails with
// "undefined reference to `hipMemcpyHtoD`" / "...HtoDAsync".
//
// See: https://github.com/CHIP-SPV/chipStar issue (OpenMM build break)

#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    hipDeviceptr_t devPtr = nullptr;
    const void* hostPtr = nullptr;
    size_t bytes = 0;

    // Synchronous variant.
    hipError_t err1 = hipMemcpyHtoD(devPtr, hostPtr, bytes);
    printf("hipMemcpyHtoD returned %d (%s)\n",
           (int)err1, hipGetErrorString(err1));

    // Async variant.
    hipStream_t stream = nullptr;
    hipError_t err2 = hipMemcpyHtoDAsync(devPtr, hostPtr, bytes, stream);
    printf("hipMemcpyHtoDAsync returned %d (%s)\n",
           (int)err2, hipGetErrorString(err2));

    // The point of this test is that the link succeeds. Any runtime
    // result is acceptable — a real implementation would reject the
    // null inputs. Return 0 unconditionally so the test passes once
    // the ABI is correct.
    return 0;
}
