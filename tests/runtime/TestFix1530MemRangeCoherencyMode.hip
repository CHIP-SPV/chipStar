// Regression test for https://github.com/CHIP-SPV/chipStar/issues/1530
//
// hipMemRangeGetAttribute(hipMemRangeAttributeCoherencyMode) must answer for
// every tracked allocation, the way ROCm does (rocclr Device::GetSvmAttributes
// reads the HSA pool flags of the pointer): device memory and non coherent
// host memory are coarse grained, coherent host memory and managed memory are
// fine grained, and hipMemAdviseSetCoarseGrain flips a managed range to
// coarse grained. Kokkos hip.memory_requirements relies on all three of its
// memory spaces (hipMalloc, hipHostMalloc(hipHostMallocNonCoherent) and
// hipMallocManaged + hipMemAdviseSetCoarseGrain) reporting coarse grain.

#include <hip/hip_runtime.h>
#include <cstdio>

static int Failures = 0;

static const char *modeName(hipMemRangeCoherencyMode Mode) {
  switch (Mode) {
  case hipMemRangeCoherencyModeFineGrain:
    return "FineGrain";
  case hipMemRangeCoherencyModeCoarseGrain:
    return "CoarseGrain";
  case hipMemRangeCoherencyModeIndeterminate:
    return "Indeterminate";
  }
  return "?";
}

static void checkMode(const char *Kind, const void *Ptr, size_t Size,
                      hipMemRangeCoherencyMode Expected) {
  hipMemRangeCoherencyMode Mode = hipMemRangeCoherencyModeIndeterminate;
  hipError_t Err = hipMemRangeGetAttribute(
      &Mode, sizeof(Mode), hipMemRangeAttributeCoherencyMode, Ptr, Size);
  if (Err != hipSuccess) {
    printf("FAIL: %s: hipMemRangeGetAttribute(CoherencyMode) returned %s, "
           "expected hipSuccess\n",
           Kind, hipGetErrorName(Err));
    Failures++;
    return;
  }
  if (Mode != Expected) {
    printf("FAIL: %s: coherency mode %s, expected %s\n", Kind, modeName(Mode),
           modeName(Expected));
    Failures++;
    return;
  }

  // The plural entry point shares the attribute switch.
  hipMemRangeCoherencyMode Mode2 = hipMemRangeCoherencyModeIndeterminate;
  void *Data[1] = {&Mode2};
  size_t DataSizes[1] = {sizeof(Mode2)};
  hipMemRangeAttribute Attrs[1] = {hipMemRangeAttributeCoherencyMode};
  Err = hipMemRangeGetAttributes(Data, DataSizes, Attrs, 1, Ptr, Size);
  if (Err != hipSuccess || Mode2 != Expected) {
    printf("FAIL: %s: hipMemRangeGetAttributes(CoherencyMode) returned %s, "
           "mode %s, expected hipSuccess and %s\n",
           Kind, hipGetErrorName(Err), modeName(Mode2), modeName(Expected));
    Failures++;
    return;
  }
  printf("OK: %s: %s\n", Kind, modeName(Mode));
}

static void checkError(const char *What, hipError_t Err, hipError_t Expected) {
  if (Err != Expected) {
    printf("FAIL: %s returned %s, expected %s\n", What, hipGetErrorName(Err),
           hipGetErrorName(Expected));
    Failures++;
  } else {
    printf("OK: %s returned %s\n", What, hipGetErrorName(Err));
  }
}

int main() {
  const size_t N = 1024;
  const size_t Size = N * sizeof(int);

  void *DevPtr = nullptr;
  checkError("hipMalloc", hipMalloc(&DevPtr, Size), hipSuccess);
  void *HostPtr = nullptr;
  checkError("hipHostMalloc(default)",
             hipHostMalloc(&HostPtr, Size, hipHostMallocDefault), hipSuccess);
  void *HostNcPtr = nullptr;
  checkError("hipHostMalloc(hipHostMallocNonCoherent)",
             hipHostMalloc(&HostNcPtr, Size, hipHostMallocNonCoherent),
             hipSuccess);
  void *ManagedPtr = nullptr;
  checkError("hipMallocManaged", hipMallocManaged(&ManagedPtr, Size),
             hipSuccess);
  if (Failures)
    return 1;

  checkMode("hipMalloc", DevPtr, Size, hipMemRangeCoherencyModeCoarseGrain);
  checkMode("hipHostMalloc(default)", HostPtr, Size,
            hipMemRangeCoherencyModeFineGrain);
  checkMode("hipHostMalloc(hipHostMallocNonCoherent)", HostNcPtr, Size,
            hipMemRangeCoherencyModeCoarseGrain);
  checkMode("hipMallocManaged", ManagedPtr, Size,
            hipMemRangeCoherencyModeFineGrain);

  // Kokkos::HIPManagedSpace: hipMallocManaged followed by
  // hipMemAdviseSetCoarseGrain must report coarse grain.
  checkError("hipMemAdvise(hipMemAdviseSetCoarseGrain)",
             hipMemAdvise(ManagedPtr, Size, hipMemAdviseSetCoarseGrain, 0),
             hipSuccess);
  checkMode("hipMallocManaged + SetCoarseGrain", ManagedPtr, Size,
            hipMemRangeCoherencyModeCoarseGrain);
  checkError("hipMemAdvise(hipMemAdviseUnsetCoarseGrain)",
             hipMemAdvise(ManagedPtr, Size, hipMemAdviseUnsetCoarseGrain, 0),
             hipSuccess);
  checkMode("hipMallocManaged + UnsetCoarseGrain", ManagedPtr, Size,
            hipMemRangeCoherencyModeFineGrain);

  // A DataSize other than sizeof(hipMemRangeCoherencyMode) is still rejected.
  hipMemRangeCoherencyMode Modes[2];
  checkError("CoherencyMode with DataSize 2*sizeof(mode) on hipMalloc",
             hipMemRangeGetAttribute(Modes, sizeof(Modes),
                                     hipMemRangeAttributeCoherencyMode, DevPtr,
                                     Size),
             hipErrorInvalidValue);
  checkError("CoherencyMode with DataSize 2*sizeof(mode) on hipMallocManaged",
             hipMemRangeGetAttribute(Modes, sizeof(Modes),
                                     hipMemRangeAttributeCoherencyMode,
                                     ManagedPtr, Size),
             hipErrorInvalidValue);

  // The prefetch attribute keeps its managed only behaviour: hipInvalidDeviceId
  // before any prefetch on managed memory, hipErrorInvalidValue elsewhere.
  int Loc = 12345;
  checkError("LastPrefetchLocation on hipMallocManaged",
             hipMemRangeGetAttribute(&Loc, sizeof(Loc),
                                     hipMemRangeAttributeLastPrefetchLocation,
                                     ManagedPtr, Size),
             hipSuccess);
  if (Loc != hipInvalidDeviceId) {
    printf("FAIL: LastPrefetchLocation on hipMallocManaged is %d, expected "
           "hipInvalidDeviceId (%d)\n",
           Loc, hipInvalidDeviceId);
    Failures++;
  }
  checkError("LastPrefetchLocation on hipMalloc",
             hipMemRangeGetAttribute(&Loc, sizeof(Loc),
                                     hipMemRangeAttributeLastPrefetchLocation,
                                     DevPtr, Size),
             hipErrorInvalidValue);

  (void)hipFree(ManagedPtr);
  (void)hipHostFree(HostNcPtr);
  (void)hipHostFree(HostPtr);
  (void)hipFree(DevPtr);

  if (Failures) {
    printf("FAILED: %d check(s)\n", Failures);
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
