// Reproduces: the message of a failed device-side assertion reaches the host
// on stdout only, while ROCm reports it on stderr. gtest death tests match
// their regex against captured stderr, so a Kokkos death test that aborts on
// the device (hip_DeathTest.abort_from_device and friends) fails with
// "died but not with expected error" even though the process did die with
// SIGABRT and the message was printed.
//
// The kernel fails an assertion the way Kokkos::abort does on the device
// (Kokkos_HIP_Abort.hpp: __assert_fail(msg, "", 0, "")). The runtime services
// the abort after the launch and terminates the process with SIGABRT.
// tests/run_device_abort_stderr_test.cmake runs this binary with stdout and
// stderr captured separately and owns the verdict: the process must die with
// SIGABRT and the assertion message must be on stderr.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void assertOnDevice() {
  __assert_fail("Meurs, pourriture communiste !", "", 0, "");
}

int main() {
  unsetenv("CHIP_HOST_IGNORES_DEVICE_ABORT"); // Force abort behaviour.
  assertOnDevice<<<1, 1>>>();
  (void)hipDeviceSynchronize();
  // Control must not reach here: the runtime aborts the process when it
  // observes the device-side abort request.
  std::printf("FAIL: the host survived a device-side abort\n");
  return 1;
}
