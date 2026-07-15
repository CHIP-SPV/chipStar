// Regression test for issue #499 (HIP error-type conformance).
//
// Per the HIP API contract, hipDeviceGetCacheConfig succeeds and simply
// ignores the cache-configuration hint on architectures without a
// reconfigurable cache (see hip_runtime_api.h: "@returns #hipSuccess,
// #hipErrorNotInitialized ... This hint is ignored on these architectures").
// chipStar previously returned hipErrorNotSupported, which deviates from
// the reference behaviour and breaks HIP tests that check error types.
//
// Note: hipDeviceSetCacheConfig legitimately returns hipErrorNotSupported
// on AMD ("This API is not implemented on AMD platform"), so this test
// only pins the GetCacheConfig behaviour.

#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  hipFuncCache_t Cfg;
  hipError_t Err = hipDeviceGetCacheConfig(&Cfg);
  if (Err != hipSuccess) {
    printf("FAIL: hipDeviceGetCacheConfig returned %d (%s), expected "
           "hipSuccess\n",
           (int)Err, hipGetErrorName(Err));
    return 1;
  }

  // A null output pointer must still be rejected with hipErrorInvalidValue.
  Err = hipDeviceGetCacheConfig(nullptr);
  if (Err != hipErrorInvalidValue) {
    printf("FAIL: hipDeviceGetCacheConfig(nullptr) returned %d (%s), expected "
           "hipErrorInvalidValue\n",
           (int)Err, hipGetErrorName(Err));
    return 1;
  }

  printf("PASS\n");
  return 0;
}
