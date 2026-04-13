// Regression test for OpenCL event-callback deadlock (#1090).
//
// The bug: CHIPQueueOpenCL::launchKernel() registered a clSetEventCallback to
// free the ArgSpillBuffer when the kernel completed.  The callback fires on an
// OpenCL-internal thread that holds OpenCL driver locks; calling
// clReleaseMemObject() from within that callback can re-enter the same lock
// and deadlock.
//
// The test launches many kernels with large struct arguments (which trigger
// ArgSpillBuffer creation) without ever synchronising between launches.  This
// maximises the chance that callbacks fire while the driver is busy, hitting
// the deadlock on unpatched code.
//
// After the fix the ArgSpillBuffer lifetime is managed through the chipStar
// event's KeepAlives list, avoiding any OpenCL API call from a callback.

#include <hip/hip_runtime.h>
#include <iostream>

// Large struct → lowered to PODByRef in SPIR-V → ArgSpillBuffer allocated.
struct BigArg {
  float data[64]; // 256 bytes
};

__global__ void kernelWithBigArg(BigArg a, float *out) {
  int tid = threadIdx.x;
  if (tid < 64)
    out[tid] = a.data[tid];
}

int main() {
  constexpr int N = 2000;
  constexpr size_t Sz = sizeof(float) * 64;

  float *d_out;
  if (hipMalloc(&d_out, Sz) != hipSuccess) {
    std::cerr << "hipMalloc failed\n";
    return 1;
  }

  // Launch many kernels without syncing between them so that multiple
  // kernelEventCallback invocations can stack up and contend.
  for (int i = 0; i < N; i++) {
    BigArg a;
    for (int j = 0; j < 64; j++)
      a.data[j] = (float)(i * 64 + j);
    hipLaunchKernelGGL(kernelWithBigArg, dim3(1), dim3(64), 0, 0, a, d_out);
  }

  if (hipDeviceSynchronize() != hipSuccess) {
    std::cerr << "hipDeviceSynchronize failed\n";
    return 2;
  }

  hipFree(d_out);
  std::cout << "PASSED\n";
  return 0;
}
