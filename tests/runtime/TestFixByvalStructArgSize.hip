// Reproduces: a small aggregate passed to a kernel BY VALUE is delivered with
// the wrong argument size on strict OpenCL drivers (e.g. rusticl/radeonsi).
//
// chipStar computes the by-value struct size without trailing padding
// (SPIRVtypeStruct, src/spirv.cc) -- e.g. {int; char;} -> 5 instead of 8 --
// and passes that size to clSetKernelArg. Strict drivers reject the mismatched
// size (CL_INVALID_ARG_SIZE), so the argument is left unset and the launch
// fails with CL_INVALID_KERNEL_ARGS; the kernel never sees the correct values.
//
// Root-caused from Unit_hipClassKernel_Value.

#include <hip/hip_runtime.h>
#include <cstdio>

struct ByValStruct {
  int i;
  char c;
};  // sizeof == 8 (4 + 1 + 3 tail padding); chipStar computes 5

__global__ void byvalKernel(ByValStruct s, int *out) {
  out[0] = (s.i == 42 && s.c == 'X') ? 1 : 0;
}

int main() {
  int *out_d = nullptr;
  if (hipMalloc(&out_d, sizeof(int)) != hipSuccess) {
    printf("FAIL: hipMalloc\n");
    return 1;
  }
  int zero = 0;
  hipMemcpy(out_d, &zero, sizeof(int), hipMemcpyHostToDevice);

  ByValStruct s;
  s.i = 42;
  s.c = 'X';
  byvalKernel<<<1, 1>>>(s, out_d);

  hipError_t launchErr = hipGetLastError();
  hipError_t syncErr = hipDeviceSynchronize();

  int out_h = -1;
  hipMemcpy(&out_h, out_d, sizeof(int), hipMemcpyDeviceToHost);
  hipFree(out_d);

  if (launchErr != hipSuccess || syncErr != hipSuccess) {
    printf("FAIL: launch=%s sync=%s\n", hipGetErrorString(launchErr),
           hipGetErrorString(syncErr));
    return 1;
  }
  if (out_h != 1) {
    printf("FAIL: kernel saw wrong by-value struct contents (out=%d)\n", out_h);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
