// hipGetNativeEventFromHipEvent must retain the cl_event it returns.
#include "backend/OpenCL/CHIPBackendOpenCL.hh"
#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  if (hipFree(nullptr) != hipSuccess)
    return 1;
  auto *Ctx = dynamic_cast<CHIPContextOpenCL *>(Backend->getActiveContext());
  if (!Ctx) {
    printf("HIP_SKIP_THIS_TEST: OpenCL only\n");
    return CHIP_SKIP_TEST;
  }
  // A user event has no command, so nothing else changes its refcount.
  cl_int Err;
  cl_event UE = clCreateUserEvent(Ctx->get()->get(), &Err);
  if (Err != CL_SUCCESS)
    return 1;
  auto *E = static_cast<chipstar::Event *>(hipGetHipEventFromNativeEvent(UE));
  E->setRecording();

  cl_uint Before, After;
  clGetEventInfo(UE, CL_EVENT_REFERENCE_COUNT, sizeof(Before), &Before, 0);
  void *N = hipGetNativeEventFromHipEvent(E);
  clGetEventInfo(UE, CL_EVENT_REFERENCE_COUNT, sizeof(After), &After, 0);
  if (N != UE || After != Before + 1) {
    printf("FAIL: export left the refcount at %u, expected %u\n", After,
           Before + 1);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
