// Minimal reproducer for chipStar issue #1181
// HeCBench lebesgue benchmark calls atomicMax() on double*.
//
// The bug: chipStar's devicelib implements atomicMax for double using two
// separate 32-bit atomic_cmpxchg operations (one for each half of the
// double). This is fundamentally non-atomic -- another thread can modify
// the value between the two 32-bit CAS operations, corrupting the result.
//
// This test creates high contention on a single double with values whose
// upper and lower 32 bits both vary, maximizing the chance of exposing
// the torn-write race. It runs multiple iterations to catch intermittent
// failures.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <hip/hip_runtime.h>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

// Reproduce the exact pattern from HeCBench lebesgue kernels.cu line 23:
//   atomicMax(lmax, t);
// where lmax is double* and t is double.
__global__ void maxKernel(double *maxVal, const double *data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicMax(maxVal, data[i]);
  }
}

int main() {
  // Use enough elements to fill many wavefronts/warps and create contention.
  const int N = 65536;
  const int NUM_ITERS = 50;
  int failures = 0;

  double *h_data = new double[N];

  double *d_data, *d_max;
  hipMalloc(&d_data, N * sizeof(double));
  hipMalloc(&d_max, sizeof(double));

  // Use a fixed seed for reproducibility.
  srand(42);

  for (int iter = 0; iter < NUM_ITERS; iter++) {
    // Generate values that stress the split-CAS bug:
    // Use doubles with varying upper and lower 32-bit halves.
    // Values near each other but not identical maximize contention
    // on both halves of the double representation.
    double cpuMax = -DBL_MAX;
    for (int i = 0; i < N; i++) {
      // Mix of large and small values with different bit patterns.
      // The key is that the maximum often changes both the high and
      // low 32-bit words of the double, which is where the split-CAS
      // race manifests.
      h_data[i] = 1000.0 + (double)(rand() % 1000000) / 1000.0;
      if (h_data[i] > cpuMax)
        cpuMax = h_data[i];
    }

    double initMax = 0.0;
    hipMemcpy(d_data, h_data, N * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_max, &initMax, sizeof(double), hipMemcpyHostToDevice);

    // Launch with enough blocks to create real contention.
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    maxKernel<<<gridSize, blockSize>>>(d_max, d_data, N);

    double gpuMax = 0.0;
    hipMemcpy(&gpuMax, d_max, sizeof(double), hipMemcpyDeviceToHost);

    if (fabs(gpuMax - cpuMax) > 1e-10) {
      printf("FAILED iter %d: GPU max = %.15g, CPU max = %.15g\n",
             iter, gpuMax, cpuMax);
      failures++;
    }
  }

  hipFree(d_data);
  hipFree(d_max);
  delete[] h_data;

  if (failures == 0) {
    printf("PASSED (%d iterations)\n", NUM_ITERS);
    return 0;
  } else {
    printf("FAILED: %d/%d iterations produced wrong results\n",
           failures, NUM_ITERS);
    return 1;
  }
}
