// Regression test for https://github.com/CHIP-SPV/chipStar/issues/2285
// hipInitFromNativeHandles must not consume the application's cl_context
// reference when the backend it created is torn down.
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <CL/cl.h>
#include <cstdio>
#include <cstring>

static unsigned refs(cl_context C) {
  cl_uint N = 0;
  clGetContextInfo(C, CL_CONTEXT_REFERENCE_COUNT, sizeof N, &N, nullptr);
  return N;
}

int main() {
  int NumHandles = 0;
  uintptr_t H[8] = {};
  if (hipGetBackendNativeHandles(0, nullptr, &NumHandles) != hipSuccess ||
      NumHandles < 5 || NumHandles > 8 ||
      hipGetBackendNativeHandles(0, H, nullptr) != hipSuccess ||
      strcmp(reinterpret_cast<const char *>(H[0]), "opencl") != 0) {
    printf("HIP_SKIP_THIS_TEST: needs the OpenCL backend\n");
    return CHIP_SKIP_TEST;
  }
  auto P = reinterpret_cast<cl_platform_id>(H[1]);
  auto D = reinterpret_cast<cl_device_id>(H[2]);

  cl_int E;
  cl_context C1 = clCreateContext(nullptr, 1, &D, nullptr, nullptr, &E);
  cl_context C2 = clCreateContext(nullptr, 1, &D, nullptr, nullptr, &E);
  cl_command_queue Q1 = clCreateCommandQueueWithProperties(C1, D, nullptr, &E);
  cl_command_queue Q2 = clCreateCommandQueueWithProperties(C2, D, nullptr, &E);
  clRetainContext(C1); // keeps C1 readable even if the bug strikes
  unsigned Before = refs(C1);

  uintptr_t N1[4] = {(uintptr_t)P, (uintptr_t)D, (uintptr_t)C1, (uintptr_t)Q1};
  uintptr_t N2[4] = {(uintptr_t)P, (uintptr_t)D, (uintptr_t)C2, (uintptr_t)Q2};
  if (hipInitFromNativeHandles(N1, 4) != hipSuccess ||
      hipInitFromNativeHandles(N2, 4) != hipSuccess) {
    printf("FAIL: hipInitFromNativeHandles failed\n");
    return 1;
  }

  unsigned After = refs(C1);
  if (Before < 2 || After != Before) {
    printf("FAIL: C1 refcount %u before adoption, %u after teardown\n", Before,
           After);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
