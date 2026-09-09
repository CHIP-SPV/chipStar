// A clFlush that fails with a spec legal error must reach the application as
// a HIP error, not kill it. https://github.com/CHIP-SPV/chipStar/issues/1604
//
// ClFlushErrorInterposer returns CL_OUT_OF_RESOURCES from one clFlush call.
// chipStar checks clFlush with CHIPERR_CHECK_LOG_AND_THROW_TABLE, which calls
// std::abort() on a status its conversion table does not map, and the clFlush
// row did not map CL_OUT_OF_RESOURCES even though OpenCL 3.0 lists it for that
// call. So before the fix the injected failure terminated the process on a
// signal; after it, the call returns hipErrorOutOfMemory and the process
// carries on.
//
// This binary is driven by tests/run_clflush_error_map_test.cmake, which
// sweeps the ordinal because how many times chipStar flushes before reaching
// this point depends on the backend and the driver. It reports what each HIP
// call returned and always exits 0: dying is the failure the driver looks for,
// so the verdict must not be confused with an exit code of this program's own.

#include <hip/hip_runtime.h>

#include <cstdio>

__global__ void bump(int *P) { *P += 1; }

int main() {
  int *P = nullptr;
  if (hipMalloc(&P, sizeof(int)) != hipSuccess) {
    printf("REPORT: hipMalloc failed\n");
    return 0;
  }
  hipError_t Err = hipMemset(P, 0, sizeof(int));
  printf("REPORT: hipMemset %s\n", hipGetErrorName(Err));

  // hipEventRecord flushes the queue it moves the stream to, and
  // hipStreamQuery flushes the queue it polls, so between them they reach the
  // flush sites this test is about.
  hipEvent_t E;
  Err = hipEventCreate(&E);
  printf("REPORT: hipEventCreate %s\n", hipGetErrorName(Err));

  bump<<<1, 1>>>(P);
  Err = hipEventRecord(E, 0);
  printf("REPORT: hipEventRecord %s\n", hipGetErrorName(Err));
  fflush(stdout);

  Err = hipStreamQuery(0);
  printf("REPORT: hipStreamQuery %s\n", hipGetErrorName(Err));
  fflush(stdout);

  // The runtime has to stay usable after it has reported the failure.
  Err = hipStreamSynchronize(0);
  printf("REPORT: hipStreamSynchronize %s\n", hipGetErrorName(Err));

  // Not checked: the injected failure may have left either of these with
  // something to report, and the point here is only that the process got
  // this far.
  (void)hipEventDestroy(E);
  (void)hipFree(P);
  printf("REPORT: survived\n");
  fflush(stdout);
  return 0;
}
