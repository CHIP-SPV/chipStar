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
//
// The kind of memory behind each pointer depends on the backend: hipMalloc is
// device memory on Level Zero and on OpenCL with Intel USM, but unified (SVM)
// memory on macOS (clvk). The expectations are therefore derived from the
// hipMemoryType hipPointerGetAttributes reports for the pointer, not from the
// allocating call.

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

static const char *typeName(hipMemoryType Type) {
  switch (Type) {
  case hipMemoryTypeHost:
    return "hipMemoryTypeHost";
  case hipMemoryTypeDevice:
    return "hipMemoryTypeDevice";
  case hipMemoryTypeManaged:
    return "hipMemoryTypeManaged";
  case hipMemoryTypeUnified:
    return "hipMemoryTypeUnified";
  default:
    return "?";
  }
}

// Managed and unified memory carry the prefetch, read mostly, preferred
// location and accessed by attributes and honour hipMemAdviseSetCoarseGrain.
static bool isManagedType(hipMemoryType Type) {
  return Type == hipMemoryTypeManaged || Type == hipMemoryTypeUnified;
}

// The memory type behind Ptr as the runtime tracks it.
static hipMemoryType pointerType(const char *Kind, const void *Ptr) {
  hipPointerAttribute_t Attr = {};
  hipError_t Err = hipPointerGetAttributes(&Attr, Ptr);
  if (Err != hipSuccess) {
    printf("FAIL: %s: hipPointerGetAttributes returned %s, expected "
           "hipSuccess\n",
           Kind, hipGetErrorName(Err));
    Failures++;
    return hipMemoryTypeUnregistered;
  }
  printf("INFO: %s is %s\n", Kind, typeName(Attr.type));
  return Attr.type;
}

// The coherency mode ROCm reports for a pointer of the given kind: device
// memory is coarse grained, host memory is fine grained unless allocated with
// hipHostMallocNonCoherent, managed and unified memory is fine grained unless
// hipMemAdviseSetCoarseGrain is in effect.
static hipMemRangeCoherencyMode expectedMode(hipMemoryType Type,
                                             bool NonCoherentHost,
                                             bool CoarseGrainAdvised) {
  switch (Type) {
  case hipMemoryTypeHost:
    return NonCoherentHost ? hipMemRangeCoherencyModeCoarseGrain
                           : hipMemRangeCoherencyModeFineGrain;
  case hipMemoryTypeManaged:
  case hipMemoryTypeUnified:
    return CoarseGrainAdvised ? hipMemRangeCoherencyModeCoarseGrain
                              : hipMemRangeCoherencyModeFineGrain;
  default:
    return hipMemRangeCoherencyModeCoarseGrain;
  }
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

// The prefetch attribute keeps its managed only behaviour: hipInvalidDeviceId
// before any prefetch on managed or unified memory, hipErrorInvalidValue on
// everything else.
static void checkPrefetchLocation(const char *Kind, const void *Ptr,
                                  size_t Size, bool Managed) {
  char What[128];
  snprintf(What, sizeof(What), "LastPrefetchLocation on %s", Kind);
  int Loc = 12345;
  hipError_t Err = hipMemRangeGetAttribute(
      &Loc, sizeof(Loc), hipMemRangeAttributeLastPrefetchLocation, Ptr, Size);
  checkError(What, Err, Managed ? hipSuccess : hipErrorInvalidValue);
  if (Managed && Err == hipSuccess && Loc != hipInvalidDeviceId) {
    printf("FAIL: %s is %d, expected hipInvalidDeviceId (%d)\n", What, Loc,
           hipInvalidDeviceId);
    Failures++;
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

  hipMemoryType DevType = pointerType("hipMalloc", DevPtr);
  hipMemoryType HostType = pointerType("hipHostMalloc(default)", HostPtr);
  hipMemoryType HostNcType =
      pointerType("hipHostMalloc(hipHostMallocNonCoherent)", HostNcPtr);
  hipMemoryType ManagedType = pointerType("hipMallocManaged", ManagedPtr);
  if (Failures)
    return 1;
  if (!isManagedType(ManagedType)) {
    printf("FAIL: hipMallocManaged memory is %s, expected managed or "
           "unified\n",
           typeName(ManagedType));
    return 1;
  }

  checkMode("hipMalloc", DevPtr, Size, expectedMode(DevType, false, false));
  checkMode("hipHostMalloc(default)", HostPtr, Size,
            expectedMode(HostType, false, false));
  checkMode("hipHostMalloc(hipHostMallocNonCoherent)", HostNcPtr, Size,
            expectedMode(HostNcType, true, false));
  checkMode("hipMallocManaged", ManagedPtr, Size,
            expectedMode(ManagedType, false, false));

  // Kokkos::HIPManagedSpace: hipMallocManaged followed by
  // hipMemAdviseSetCoarseGrain must report coarse grain.
  checkError("hipMemAdvise(hipMemAdviseSetCoarseGrain)",
             hipMemAdvise(ManagedPtr, Size, hipMemAdviseSetCoarseGrain, 0),
             hipSuccess);
  checkMode("hipMallocManaged + SetCoarseGrain", ManagedPtr, Size,
            expectedMode(ManagedType, false, true));
  checkError("hipMemAdvise(hipMemAdviseUnsetCoarseGrain)",
             hipMemAdvise(ManagedPtr, Size, hipMemAdviseUnsetCoarseGrain, 0),
             hipSuccess);
  checkMode("hipMallocManaged + UnsetCoarseGrain", ManagedPtr, Size,
            expectedMode(ManagedType, false, false));

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

  checkPrefetchLocation("hipMallocManaged", ManagedPtr, Size, true);
  checkPrefetchLocation("hipMalloc", DevPtr, Size, isManagedType(DevType));
  checkPrefetchLocation("hipHostMalloc(default)", HostPtr, Size,
                        isManagedType(HostType));

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
