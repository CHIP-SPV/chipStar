// Reproduces: device atomics on hipMallocManaged memory give wrong results on
// Intel Data Center GPU Max (PVC) under Level Zero. chipStar backs managed
// memory with zeMemAllocHost, and on PVC host USM reports no ATOMIC access
// capability (hostAllocCapabilities = RW): a 32-bit atomicAdd silently leaves
// the counter at 0 and a 64-bit atomic raises AtomicAccessViolation (NEO
// aborts). Kokkos HIPManagedSpace (hipMallocManaged + hipMemAdvise
// SetCoarseGrain) and its ScatterView test hit exactly this.
// https://github.com/CHIP-SPV/chipStar/issues/1489
//
// The 32-bit and 64-bit kernels run separately, with stdout flushed in
// between, so the log shows the 32-bit mismatch before a 64-bit fault kills
// the process. Run with CHIP_LOGLEVEL=info to see which USM kind the runtime
// picked for managed memory.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(Expr)                                                            \
  do {                                                                         \
    hipError_t E = (Expr);                                                     \
    if (E != hipSuccess) {                                                     \
      printf("FAIL: %s -> %s (%d) at line %d\n", #Expr, hipGetErrorString(E), \
             (int)E, __LINE__);                                                \
      return 1;                                                                \
    }                                                                          \
  } while (0)

constexpr unsigned NumBlocks = 64;
constexpr unsigned NumThreads = 256;
constexpr unsigned NumSlots = 4;
constexpr unsigned long long ExpectedPerSlot =
    (unsigned long long)NumBlocks * NumThreads / NumSlots;

struct Counters32 {
  unsigned int Add[NumSlots];
};

struct Counters64 {
  unsigned long long Add[NumSlots];
  unsigned long long Min;
  unsigned long long Max;
};

__global__ void atomics32(Counters32 *C) {
  unsigned Tid = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(&C->Add[Tid % NumSlots], 1u);
}

__global__ void atomics64(Counters64 *C) {
  unsigned Tid = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(&C->Add[Tid % NumSlots], 1ull);
  atomicMin(&C->Min, (unsigned long long)Tid + 1000);
  atomicMax(&C->Max, (unsigned long long)Tid + 1000);
}

static int Failures = 0;

static void expectU64(const char *What, unsigned long long Got,
                      unsigned long long Want) {
  if (Got != Want) {
    printf("  mismatch: %s = %llu, expected %llu\n", What, Got, Want);
    Failures++;
  }
}

int main() {
  hipDeviceProp_t Props;
  CHECK(hipGetDeviceProperties(&Props, 0));
  printf("device: %s\n", Props.name);
  printf("managedMemory=%d concurrentManagedAccess=%d "
         "hostNativeAtomicSupported=%d\n",
         Props.managedMemory, Props.concurrentManagedAccess,
         Props.hostNativeAtomicSupported);
  if (!Props.managedMemory) {
    printf("HIP_SKIP_THIS_TEST: device has no managed memory\n");
    return 0;
  }

  Counters32 *C32 = nullptr;
  Counters64 *C64 = nullptr;
  CHECK(hipMallocManaged(&C32, sizeof(Counters32)));
  CHECK(hipMallocManaged(&C64, sizeof(Counters64)));

  hipPointerAttribute_t Attr;
  CHECK(hipPointerGetAttributes(&Attr, C32));
  printf("hipPointerGetAttributes: type=%d isManaged=%d\n", (int)Attr.type,
         Attr.isManaged);
  if (Attr.type != hipMemoryTypeUnified && Attr.type != hipMemoryTypeManaged) {
    printf("  managed allocation reported as memory type %d\n",
           (int)Attr.type);
    Failures++;
  }

  // Mirror what Kokkos HIPManagedSpace does after hipMallocManaged.
  CHECK(hipMemAdvise(C32, sizeof(Counters32), hipMemAdviseSetCoarseGrain, 0));
  CHECK(hipMemAdvise(C64, sizeof(Counters64), hipMemAdviseSetCoarseGrain, 0));

  // Host-side initialization through the managed pointer.
  std::memset(C32, 0, sizeof(Counters32));
  std::memset(C64, 0, sizeof(Counters64));
  C64->Min = ~0ull;

  // 32-bit atomics.
  CHECK(hipMemPrefetchAsync(C32, sizeof(Counters32), 0, 0));
  atomics32<<<NumBlocks, NumThreads>>>(C32);
  CHECK(hipGetLastError());
  CHECK(hipMemPrefetchAsync(C32, sizeof(Counters32), hipCpuDeviceId, 0));
  CHECK(hipDeviceSynchronize());

  Counters32 H32;
  CHECK(hipMemcpy(&H32, C32, sizeof(Counters32), hipMemcpyDeviceToHost));
  printf("u32 atomicAdd: copied back %u %u %u %u, host read %u %u %u %u, "
         "expected %llu each\n",
         H32.Add[0], H32.Add[1], H32.Add[2], H32.Add[3], C32->Add[0],
         C32->Add[1], C32->Add[2], C32->Add[3], ExpectedPerSlot);
  fflush(stdout);
  for (unsigned I = 0; I < NumSlots; I++) {
    expectU64("u32 add (memcpy)", H32.Add[I], ExpectedPerSlot);
    expectU64("u32 add (host read)", C32->Add[I], ExpectedPerSlot);
  }

  // 64-bit atomics.
  CHECK(hipMemPrefetchAsync(C64, sizeof(Counters64), 0, 0));
  atomics64<<<NumBlocks, NumThreads>>>(C64);
  CHECK(hipGetLastError());
  CHECK(hipMemPrefetchAsync(C64, sizeof(Counters64), hipCpuDeviceId, 0));
  CHECK(hipDeviceSynchronize());

  Counters64 H64;
  CHECK(hipMemcpy(&H64, C64, sizeof(Counters64), hipMemcpyDeviceToHost));
  printf("u64 atomicAdd: copied back %llu %llu %llu %llu, host read %llu %llu "
         "%llu %llu, expected %llu each\n",
         H64.Add[0], H64.Add[1], H64.Add[2], H64.Add[3], C64->Add[0],
         C64->Add[1], C64->Add[2], C64->Add[3], ExpectedPerSlot);
  printf("u64 atomicMin/Max: %llu %llu\n", H64.Min, H64.Max);
  fflush(stdout);
  const unsigned long long LastTid = NumBlocks * NumThreads - 1;
  for (unsigned I = 0; I < NumSlots; I++) {
    expectU64("u64 add (memcpy)", H64.Add[I], ExpectedPerSlot);
    expectU64("u64 add (host read)", C64->Add[I], ExpectedPerSlot);
  }
  expectU64("u64 min", H64.Min, 1000ull);
  expectU64("u64 max", H64.Max, LastTid + 1000);

  CHECK(hipFree(C32));
  CHECK(hipFree(C64));

  if (Failures) {
    printf("FAIL: %d mismatches\n", Failures);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
