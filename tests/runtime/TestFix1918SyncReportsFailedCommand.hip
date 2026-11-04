// hipStreamSynchronize and hipDeviceSynchronize must report a command that
// terminated abnormally on the OpenCL backend.
// https://github.com/CHIP-SPV/chipStar/issues/1918
//
// The failed command is a barrier waiting on a user event set to a negative
// status, which the OpenCL spec says terminates the barrier. A marker
// enqueued after it has completed must not hide it.
//
// Modes: stream and device fail S and synchronize it with
// hipStreamSynchronize or hipDeviceSynchronize; blocking fails the null
// stream and synchronizes S, which waits for the null stream.

#include <hip/hip_runtime.h>
#include "backend/OpenCL/CHIPBackendOpenCL.hh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int ArgC, char *ArgV[]) {
  const char *Mode = ArgC > 1 ? ArgV[1] : "stream";
  bool Device = !strcmp(Mode, "device");
  hipStream_t S;
  if (hipStreamCreate(&S) != hipSuccess) {
    printf("FAIL: hipStreamCreate\n");
    return 1;
  }
  if (!dynamic_cast<CHIPBackendOpenCL *>(::Backend)) {
    printf("HIP_SKIP_THIS_TEST: not running on OpenCL\n");
    return CHIP_SKIP_TEST;
  }
  auto *Q = strcmp(Mode, "blocking")
                ? static_cast<chipstar::Queue *>(S)
                : ::Backend->getActiveDevice()->getLegacyDefaultQueue();
  auto *Ctx = static_cast<CHIPContextOpenCL *>(Q->getContext());
  cl_int Err;
  cl_event U = clCreateUserEvent(Ctx->get()->get(), &Err);
  if (Err != CL_SUCCESS) {
    printf("FAIL: clCreateUserEvent returned %d\n", Err);
    return 1;
  }
  auto Barrier = Q->enqueueBarrier({std::make_shared<CHIPEventOpenCL>(Ctx, U)});
  Err = clSetUserEventStatus(U, -1);
  if (Err != CL_SUCCESS) {
    printf("FAIL: clSetUserEventStatus returned %d\n", Err);
    fflush(stdout);
    std::_Exit(1); // The barrier never completes, so skip teardown.
  }
  clReleaseEvent(U);
  clWaitForEvents(1, &static_cast<CHIPEventOpenCL *>(Barrier.get())->ClEvent);
  Q->enqueueMarker();

  hipError_t SyncErr = Device ? hipDeviceSynchronize() : hipStreamSynchronize(S);
  if (SyncErr != hipErrorLaunchFailure) {
    printf("FAIL: %s returned %s\n",
           Device ? "hipDeviceSynchronize" : "hipStreamSynchronize",
           hipGetErrorName(SyncErr));
    return 1;
  }
  // Checked without another clFinish: the spec leaves the queue
  // implementation-defined after a terminated command.
  if (Q->takeCommandError() != hipSuccess) {
    printf("FAIL: the failure is reported again\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}
