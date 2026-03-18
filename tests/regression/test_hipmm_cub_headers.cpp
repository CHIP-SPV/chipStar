// Reproducer for hipMM CUDA CUB header build failure.
//
// hipMM uses rocThrust, which installs CUDA Thrust compatibility
// headers that transitively #include CUDA CUB headers like
// cub/util_namespace.cuh. These don't exist in the chipStar
// environment, causing 12 hipMM tests to fail to build.
//
// This test verifies that the CUB stub headers are available
// so that rocThrust's CUDA compatibility layer can be used.
//
// Build: hipcc -x hip test_hipmm_cub_headers.cpp -I<rocthrust>/include
// Expected: compiles and prints PASS
// Bug: fails with "cub/util_namespace.cuh: No such file or directory"

#include <cstdio>

// These are the headers that hipMM pulls in transitively via rocThrust
// CUDA Thrust compatibility. They need CUB stubs to exist.
#if __has_include(<cub/util_namespace.cuh>)
#include <cub/util_namespace.cuh>
#define CUB_FOUND 1
#else
#define CUB_FOUND 0
#endif

#if __has_include(<cuda/std/type_traits>)
#include <cuda/std/type_traits>
#define CUDA_STD_FOUND 1
#else
#define CUDA_STD_FOUND 0
#endif

int main() {
    printf("CUB headers available: %s\n", CUB_FOUND ? "yes" : "no");
    printf("CUDA std headers available: %s\n", CUDA_STD_FOUND ? "yes" : "no");

    if (CUB_FOUND && CUDA_STD_FOUND) {
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL — missing stub headers needed by rocThrust CUDA compat layer\n");
        return 1;
    }
}
