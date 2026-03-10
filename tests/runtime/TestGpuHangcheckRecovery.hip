/// \file
/// Regression test for chipStar issue #1191: hipDeviceSynchronize hangs
/// indefinitely when GPU hang detection (i915 hangcheck) kills long-running
/// batches and chipStar doesn't recover.
///
/// On Intel dGPUs with numQueues=1, all HIP streams serialize on a single
/// hardware queue. When the total GPU wall time exceeds the i915 hangcheck
/// threshold (~18-20s with default heartbeat_interval_ms=2500), the GPU
/// resets all in-flight work. chipStar's finish() calls
/// zeCommandListHostSynchronize with UINT64_MAX timeout, which never
/// returns after a GPU reset.
///
/// This test launches enough long-running kernels to exceed the hangcheck.
/// EXPECTED: chipStar should detect the GPU reset and return an error
/// (not hang forever).

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void spin_kernel(long *d_out, long iterations) {
  long val = 0;
  for (long i = 0; i < iterations; i++)
    val += i % 3;
  d_out[0] = val;
}

int main() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);

  // Each kernel runs ~4s at clockRate=2.4GHz.
  // 5 kernels × 4s = 20s, exceeding the ~18s hangcheck threshold.
  const int nkernels = 5;
  const long clock_count = (long)(20.0f * prop.clockRate);

  long *d_buf;
  hipError_t err = hipMalloc(&d_buf, nkernels * sizeof(long));
  if (err != hipSuccess) {
    printf("HIP_SKIP_THIS_TEST\n");
    return 77;
  }

  hipStream_t streams[nkernels];
  for (int i = 0; i < nkernels; i++)
    hipStreamCreate(&streams[i]);

  for (int i = 0; i < nkernels; i++)
    spin_kernel<<<1, 1, 0, streams[i]>>>(&d_buf[i], clock_count);

  err = hipDeviceSynchronize();

  // If we get here (not hung), the test passes.
  // Currently chipStar hangs here, so this test is expected to FAIL
  // (timeout) until the fix is implemented.
  if (err == hipSuccess) {
    printf("PASS (all kernels completed)\n");
  } else {
    // GPU reset detected — chipStar correctly reported an error
    printf("PASS (GPU reset detected: %s)\n", hipGetErrorString(err));
  }

  for (int i = 0; i < nkernels; i++)
    hipStreamDestroy(streams[i]);
  hipFree(d_buf);
  return 0;
}
