#include <hip/hip_runtime.h>
#include <omp.h>
#include <cstdio>

// Regression test: a single translation unit mixing a HIP device kernel and
// host (CPU) OpenMP must compile and link.
__global__ void k() {
}

int main() {
  k<<<1, 1>>>(); // empty kernel
  hipDeviceSynchronize();
#pragma omp parallel num_threads(2)
  if (omp_get_thread_num() == 0)
    printf("threads: %d\n", omp_get_num_threads());

  return 0;
}
