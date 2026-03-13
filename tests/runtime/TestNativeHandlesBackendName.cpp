// Test that hipGetBackendNativeHandles returns a concrete backend name
// ("opencl" or "level0"), never "default", even when CHIP_BE is unset or set
// to "default".
//
// Regression test for https://github.com/CHIP-SPV/chipStar/issues/1199
// When CHIP_BE=default (or unset), the interop API was returning "default" as
// the backend name in NativeHandles[0], causing MKLShim to abort with
// "Unsupported backend: default".

#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

int main() {
  // Verify initialization succeeds.
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  if (err == hipErrorInitializationError || err == hipErrorNoDevice) {
    printf("HIP_SKIP_THIS_TEST: no backend available (error %d)\n", err);
    return CHIP_SKIP_TEST;
  }
  if (err != hipSuccess || count < 1) {
    printf("FAIL: hipGetDeviceCount returned %d, count=%d\n", err, count);
    return 1;
  }

  // First call: get the number of native handles.
  int numHandles = 0;
  int ret = hipGetBackendNativeHandles(0, nullptr, &numHandles);
  if (ret != hipSuccess) {
    printf("FAIL: hipGetBackendNativeHandles(count) returned %d\n", ret);
    return 1;
  }
  if (numHandles < 1) {
    printf("FAIL: hipGetBackendNativeHandles returned numHandles=%d\n",
           numHandles);
    return 1;
  }

  // Second call: get the actual handles.
  uintptr_t handles[numHandles];
  ret = hipGetBackendNativeHandles(0, handles, nullptr);
  if (ret != hipSuccess) {
    printf("FAIL: hipGetBackendNativeHandles(handles) returned %d\n", ret);
    return 1;
  }

  // NativeHandles[0] must be a pointer to the backend name string.
  const char *backendName = reinterpret_cast<const char *>(handles[0]);
  if (backendName == nullptr) {
    printf("FAIL: NativeHandles[0] is null\n");
    return 1;
  }

  printf("INFO: backend name from NativeHandles[0] = \"%s\"\n", backendName);

  if (strcmp(backendName, "opencl") != 0 && strcmp(backendName, "level0") != 0) {
    printf("FAIL: NativeHandles[0] is \"%s\", expected \"opencl\" or "
           "\"level0\" (never \"default\")\n",
           backendName);
    return 1;
  }

  printf("PASS: NativeHandles[0] = \"%s\"\n", backendName);
  return 0;
}
