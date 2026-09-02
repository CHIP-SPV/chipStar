// Reproducer for chipStar issue #1508: a `volatile` load of global memory is
// emitted as OpLoad carrying only the Volatile memory operand, which says
// nothing about caching, so on an Intel GPU with a per-Xe-core L1 (Xe-HPG,
// Xe-HPC, Xe2: DG2, Arc, PVC, B-series) the load is served from that core's
// stale L1 and never observes another work-group's store.
//
// Measured with ocloc on the module chipStar produces, a volatile load and a
// plain load compile to the SAME message on PVC (`load.ugm.d32x1t.a64`,
// desc 0x2108580); only the !nontemporal marking this pass adds changes it
// (`.uc.uc`, desc 0x2128580). CUDA compiles the same source to ld.volatile,
// which the PTX ISA defines as a relaxed system-scope operation, and HIP code
// written for that (Kokkos::volatile_load in the UnorderedMap insert list walk)
// reads stale data without the marking.
//
// This is a message-passing litmus test: the Kokkos insert idiom with the race
// removed, so it fails deterministically rather than statistically.
//
//   writer WG : plain stores to data[], __threadfence(), publish flag = 1
//   reader WGs: PRIME the cache by reading data[idx] BEFORE the write,
//               spin until flag != 0, then RE-READ data[idx]
//
// The prime is what makes it 100%: the reader is guaranteed to be holding the
// pre-write line when the writer runs. Staleness surfaces two ways, both
// counted: TIMEOUT (never observed the flag flip) and STALE (observed the flag,
// then read pre-write data).
//
// Self-validating. Phase 1 runs the identical kernel and geometry with relaxed
// atomics, which bypass a private cache by construction. If that phase is not
// clean the device cannot host the experiment (no forward-progress guarantee
// for the spin, or the work-groups were not co-resident) and the test skips
// instead of reporting a failure. Only a clean control makes a volatile-phase
// failure attributable to caching.
//
// Phase 3 is a separate regression guard, for the atomic lowering an earlier
// version of this pass used and that had to be reverted: IGC implements
// OpAtomicLoad as atomic_or(p, 0) and OpAtomicStore as an atomic exchange, and
// a Level Zero device may report an allocation kind without
// ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC (PVC does, for host allocations), where any
// GPU atomic faults the context. The volatile accesses must therefore stay
// non-atomic on page-locked and managed memory.
//
// Devices whose global loads are already coherent across cores (Gen12LP iGPUs
// such as UHD 730/770, CPU runtimes, Mali, AMD) pass every phase before and
// after the fix, which is correct: there is nothing there to fix.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(Cmd)                                                             \
  do {                                                                         \
    hipError_t Err = (Cmd);                                                    \
    if (Err != hipSuccess) {                                                   \
      printf("FAILED: %s at %s:%d\n", hipGetErrorString(Err), __FILE__,        \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static constexpr unsigned N = 4096;
static constexpr unsigned TPB = 256;
static constexpr unsigned WG = 8; // 1 writer + 7 reader work-groups
static constexpr unsigned Iters = 20;
static constexpr unsigned MaxSpin = 2000000u;
static constexpr unsigned Magic = 0xABCD0000u;

enum Mode { ModeVolatile = 0, ModeAtomic = 1 };

__device__ __forceinline__ unsigned loadOne(unsigned *P, int M) {
  if (M == ModeVolatile)
    return *reinterpret_cast<volatile unsigned *>(P);
  return __hip_atomic_load(P, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void storeOne(unsigned *P, unsigned V, int M) {
  if (M == ModeVolatile)
    *reinterpret_cast<volatile unsigned *>(P) = V;
  else
    __hip_atomic_store(P, V, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__global__ void litmus(unsigned *Data, unsigned *Flag, unsigned *Stale,
                       unsigned *Timeouts, unsigned *LatePrimed, int M) {
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      for (unsigned I = 0; I < N; ++I)
        Data[I] = Magic + I; // plain stores, as Kokkos publishes its key
      __threadfence();       // Kokkos::memory_fence()
      storeOne(Flag, 1u, M); // publish
    }
    return;
  }
  unsigned Idx = (blockIdx.x * blockDim.x + threadIdx.x) % N;
  if (loadOne(&Data[Idx], M) == Magic + Idx) {
    // The writer got there first, so this sample never held a stale line and
    // cannot demonstrate anything. Counted, not failed.
    atomicAdd(LatePrimed, 1u);
    return;
  }
  unsigned Spins = 0;
  while (loadOne(Flag, M) == 0u && ++Spins < MaxSpin) {
  }
  if (loadOne(Flag, M) == 0u) {
    atomicAdd(Timeouts, 1u);
    return;
  }
  if (loadOne(&Data[Idx], M) != Magic + Idx)
    atomicAdd(Stale, 1u);
}

// The regression guard's kernel: a volatile read-modify-write, which must stay
// non-atomic so it does not fault on an allocation kind that bans atomics.
__global__ void volatileHostAccess(volatile int *Flag) { *Flag = *Flag + 1; }

struct Result {
  unsigned long long Stale = 0, Timeouts = 0, LatePrimed = 0, Readers = 0;
  unsigned long long bad() const { return Stale + Timeouts; }
};

static Result runPhase(int M) {
  unsigned *Data, *Flag, *Stale, *Timeouts, *LatePrimed;
  CHECK(hipMalloc(&Data, N * sizeof(unsigned)));
  CHECK(hipMalloc(&Flag, sizeof(unsigned)));
  CHECK(hipMalloc(&Stale, sizeof(unsigned)));
  CHECK(hipMalloc(&Timeouts, sizeof(unsigned)));
  CHECK(hipMalloc(&LatePrimed, sizeof(unsigned)));
  Result R;
  for (unsigned It = 0; It < Iters; ++It) {
    CHECK(hipMemset(Data, 0, N * sizeof(unsigned)));
    CHECK(hipMemset(Flag, 0, sizeof(unsigned)));
    CHECK(hipMemset(Stale, 0, sizeof(unsigned)));
    CHECK(hipMemset(Timeouts, 0, sizeof(unsigned)));
    CHECK(hipMemset(LatePrimed, 0, sizeof(unsigned)));
    litmus<<<WG, TPB>>>(Data, Flag, Stale, Timeouts, LatePrimed, M);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    unsigned S, T, P;
    CHECK(hipMemcpy(&S, Stale, sizeof(unsigned), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&T, Timeouts, sizeof(unsigned), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&P, LatePrimed, sizeof(unsigned), hipMemcpyDeviceToHost));
    R.Stale += S;
    R.Timeouts += T;
    R.LatePrimed += P;
  }
  R.Readers = (unsigned long long)(WG - 1) * TPB * Iters;
  CHECK(hipFree(Data));
  CHECK(hipFree(Flag));
  CHECK(hipFree(Stale));
  CHECK(hipFree(Timeouts));
  CHECK(hipFree(LatePrimed));
  return R;
}

// Seed and read back through hipMemcpy, not a plain host store: on devices
// whose OpenCL SVM support is coarse grained only, managed allocations are
// clSVMAlloc buffers whose contents are undefined outside a map / unmap pair
// (chipStar issue #1535, unrelated to this pass). The allocation kind is the
// point here; the host's route to it is incidental.
static int runOneHostVisibleCase(const char *Kind, int *Flag) {
  int Seed = 41, Got = -1;
  CHECK(hipMemcpy(Flag, &Seed, sizeof(int), hipMemcpyHostToDevice));
  volatileHostAccess<<<1, 1>>>(Flag);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(&Got, Flag, sizeof(int), hipMemcpyDeviceToHost));
  if (Got != 42) {
    printf("guard: %s flag = %d, expected 42\n", Kind, Got);
    return 1;
  }
  return 0;
}

static int runHostVisibleGuard() {
  int Errors = 0;
  int *Pinned = nullptr;
  CHECK(hipHostMalloc(&Pinned, sizeof(int)));
  Errors += runOneHostVisibleCase("pinned", Pinned);
  CHECK(hipHostFree(Pinned));

  int Device = 0;
  CHECK(hipGetDevice(&Device));
  int ManagedSupport = 0;
  CHECK(hipDeviceGetAttribute(&ManagedSupport, hipDeviceAttributeManagedMemory,
                              Device));
  if (ManagedSupport) {
    int *Managed = nullptr;
    CHECK(hipMallocManaged(&Managed, sizeof(int)));
    Errors += runOneHostVisibleCase("managed", Managed);
    CHECK(hipFree(Managed));
  }
  return Errors;
}

static void report(const char *Name, const Result &R) {
  printf("%-9s readers=%llu stale=%llu timeouts=%llu late-primed=%llu\n", Name,
         R.Readers, R.Stale, R.Timeouts, R.LatePrimed);
}

int main(int argc, char **argv) {
  // Phase selection so an abort in one phase cannot mask another's result.
  //   (none) = all phases, "staleness" = phases 1+2 only, "guard" = phase 3 only
  const char *Sel = argc > 1 ? argv[1] : "all";
  const bool DoStale = !strcmp(Sel, "all") || !strcmp(Sel, "staleness");
  const bool DoGuard = !strcmp(Sel, "all") || !strcmp(Sel, "guard");
  if (!DoStale && !DoGuard) { printf("usage: %s [all|staleness|guard]\n", argv[0]); return 2; }
  if (!DoStale) {
    int GE = runHostVisibleGuard();
    printf("guard:    pinned/managed volatile RMW errors=%d\n", GE);
    if (GE) { printf("FAILED: volatile accesses did not behave as plain non-atomic accesses on host-visible memory (%d errors)\n", GE); return 1; }
    printf("PASSED\n"); return 0;
  }

  // Phase 1: the control.
  Result Control = runPhase(ModeAtomic);
  report("atomic:", Control);
  if (Control.bad()) {
    printf("HIP_SKIP_THIS_TEST: the atomic control did not complete cleanly "
           "(%llu of %llu readers), so this device cannot host the probe\n",
           Control.bad(), Control.Readers);
    return 0;
  }
  if (Control.LatePrimed == Control.Readers) {
    printf("HIP_SKIP_THIS_TEST: every reader was late-primed, so no sample held "
           "a stale line\n");
    return 0;
  }

  // Phase 2: the subject. Identical kernel and geometry, volatile accesses.
  Result Subject = runPhase(ModeVolatile);
  report("volatile:", Subject);

  // Phase 3: the non-atomic guard on host-visible allocations.
  int GuardErrors = 0;
  if (DoGuard) {
    GuardErrors = runHostVisibleGuard();
    printf("guard:    pinned/managed volatile RMW errors=%d\n", GuardErrors);
  }

  if (Subject.bad()) {
    printf("FAILED: %llu of %llu volatile readers did not observe the writer's "
           "store (%llu stale reads, %llu flag timeouts) while the atomic "
           "control was clean. A volatile global load is being served from a "
           "stale core-private cache: chipStar issue #1508.\n",
           Subject.bad(), Subject.Readers, Subject.Stale, Subject.Timeouts);
    return 1;
  }
  if (GuardErrors) {
    printf("FAILED: the volatile accesses did not behave as plain non-atomic "
           "accesses on host-visible memory (%d errors)\n", GuardErrors);
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
