// #1996: a plain OpenCL kernel launch must not request a cl_event.

#include <hip/hip_runtime.h>
#include "backend/OpenCL/CHIPBackendOpenCL.hh"

#include <dlfcn.h>
#include <cstdio>

static int NumLaunches = 0, NumLaunchEvents = 0;

// libCHIP references this symbol, so the executable's definition interposes.
extern "C" cl_int clEnqueueNDRangeKernel(cl_command_queue Q, cl_kernel K,
                                         cl_uint Dims, const size_t *Offset,
                                         const size_t *Global,
                                         const size_t *Local, cl_uint NumWait,
                                         const cl_event *Wait, cl_event *Ev) {
  using FnTy = decltype(&clEnqueueNDRangeKernel);
  static auto Real = (FnTy)dlsym(RTLD_NEXT, "clEnqueueNDRangeKernel");
  NumLaunches++;
  NumLaunchEvents += Ev != nullptr;
  return Real(Q, K, Dims, Offset, Global, Local, NumWait, Wait, Ev);
}

__global__ void k(int *P) { *P = 1; }

int main() {
  int *P;
  if (hipMalloc(&P, sizeof(int)) != hipSuccess)
    return 1;
  if (!dynamic_cast<CHIPBackendOpenCL *>(Backend)) {
    printf("HIP_SKIP_THIS_TEST: not running on OpenCL\n");
    return CHIP_SKIP_TEST;
  }
  // The first launch also runs chipStar's own setup kernels; count after it.
  k<<<1, 1>>>(P);
  if (hipDeviceSynchronize() != hipSuccess)
    return 1;
  NumLaunches = NumLaunchEvents = 0;
  for (int I = 0; I < 10; I++)
    k<<<1, 1>>>(P);
  if (hipDeviceSynchronize() != hipSuccess || hipGetLastError() != hipSuccess)
    return 1;

  printf("%d launches, %d with an event\n", NumLaunches, NumLaunchEvents);
  if (NumLaunches != 10) {
    printf("FAIL: the interposer saw %d of 10 launches\n", NumLaunches);
    return 1;
  }
  if (NumLaunchEvents) {
    printf("FAIL: plain launches requested a cl_event\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}
