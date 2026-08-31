// Reproducer: volatile loads and stores in global memory reach the SPIR-V
// producer as plain `load volatile` / `store volatile` and end up as OpLoad /
// OpStore with the Volatile memory operand, which says nothing about caching,
// so IGC serves them from the Xe core's L1. CUDA compiles the same source to
// ld.volatile / st.volatile, which the PTX ISA (8.4.2 "volatile Operation")
// defines as relaxed memory operations at system scope, and code written
// against that (Kokkos::volatile_load in the UnorderedMap insert list walk)
// reads stale L1 data on PVC. chipStar marks them !nontemporal, which the
// SPIR-V producers emit as the Nontemporal memory operand and IGC maps to an
// L1 uncached access.
//
// This executable checks that the accesses still compute the right values, and
// that they work on page-locked host and managed memory: the marking must not
// turn them into atomics, because a Level Zero device may report an allocation
// kind without ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC (PVC does, for host
// allocations) and then faults the GPU on any atomic to it.
//
// TestFixVolatileLoadLoweringSPIRV.bash compiles this file with --save-temps and
// inspects the lowered device bitcode (and the SPIR-V module) for the marking;
// that is the check that fails without it.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr int NumBlocks = 4;
static constexpr int BlockSize = 256;
static constexpr int N = NumBlocks * BlockSize;

#define CHECK(Cmd)                                                             \
  do {                                                                         \
    hipError_t Err = (Cmd);                                                    \
    if (Err != hipSuccess) {                                                   \
      printf("FAILED: %s at %s:%d\n", hipGetErrorString(Err), __FILE__,        \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// The accesses the marking must reach: volatile loads and stores of every
// width through global memory pointers.
__global__ void volatileAccess(const unsigned *In32,
                               const unsigned long long *In64,
                               const unsigned short *In16, unsigned *Out32,
                               unsigned long long *Out64,
                               unsigned short *Out16) {
  int Tid = blockIdx.x * blockDim.x + threadIdx.x;
  const volatile unsigned *VIn32 = In32;
  const volatile unsigned long long *VIn64 = In64;
  const volatile unsigned short *VIn16 = In16;
  volatile unsigned *VOut32 = Out32;
  volatile unsigned long long *VOut64 = Out64;
  volatile unsigned short *VOut16 = Out16;
  unsigned A = VIn32[Tid];
  unsigned long long B = VIn64[Tid];
  VOut32[Tid] = A + 1u;
  VOut64[Tid] = B + 1ull;
  if (Tid < BlockSize)
    VOut16[Tid] = VIn16[Tid] + 1;
}

// Accesses the marking must leave alone: work-group local memory, which never
// leaves the core, so there is no other core's cache to get past.
__global__ void volatileLocal(unsigned *OutLocal) {
  __shared__ unsigned Scratch[BlockSize];
  int Tid = threadIdx.x;
  volatile unsigned *VScratch = Scratch;
  VScratch[Tid] = Tid;
  __syncthreads();
  OutLocal[Tid] = VScratch[(Tid + 1) % BlockSize];
}

// The regression guard for the marking staying non-atomic: this runs on
// page-locked host and on managed memory, which a Level Zero device may refuse
// atomics on.
__global__ void volatileHostAccess(volatile int *Flag) { *Flag = *Flag + 1; }

// Runs the volatile read-modify-write on one host-visible allocation and
// reports whether the device saw and published the right value.
//
// The seed and the readback go through hipMemcpy rather than a plain host store
// and load on purpose. What this checks is that the *device* access stayed
// non-atomic, so the allocation kind is the whole point and the host's route to
// it is incidental. Touching the pointer directly from the host would drag in a
// second, unrelated promise: on devices whose OpenCL SVM support is coarse
// grained only (Mali G52, rusticl on the W6400) chipStar's managed allocations
// are clSVMAlloc buffers whose contents are undefined outside a map / unmap
// pair, so a direct host store there is silently lost. That is chipStar issue
// https://github.com/CHIP-SPV/chipStar/issues/1535 and it has nothing to do
// with this pass. hipMemcpy is defined on every backend and keeps the guard
// running everywhere instead of skipping on the devices that need it least.
static int runOneHostVisibleCase(const char *Kind, int *Flag) {
  int Seed = 41, Got = -1;
  CHECK(hipMemcpy(Flag, &Seed, sizeof(int), hipMemcpyHostToDevice));
  volatileHostAccess<<<1, 1>>>(Flag);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(&Got, Flag, sizeof(int), hipMemcpyDeviceToHost));
  if (Got != 42) {
    printf("%s flag = %d, expected 42\n", Kind, Got);
    return 1;
  }
  return 0;
}

static int runOnHostVisibleMemory() {
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
  if (!ManagedSupport)
    return Errors;

  int *Managed = nullptr;
  CHECK(hipMallocManaged(&Managed, sizeof(int)));
  Errors += runOneHostVisibleCase("managed", Managed);
  CHECK(hipFree(Managed));
  return Errors;
}

int main() {
  std::vector<unsigned> HIn32(N), HOut32(N), HOutLocal(BlockSize);
  std::vector<unsigned long long> HIn64(N), HOut64(N);
  std::vector<unsigned short> HIn16(BlockSize), HOut16(BlockSize);
  for (int I = 0; I < N; ++I) {
    HIn32[I] = 0x10000u + I;
    HIn64[I] = 0x100000000ull + I;
  }
  for (int I = 0; I < BlockSize; ++I)
    HIn16[I] = 0x100 + I;

  unsigned *In32, *Out32, *OutLocal;
  unsigned long long *In64, *Out64;
  unsigned short *In16, *Out16;
  CHECK(hipMalloc(&In32, N * sizeof(unsigned)));
  CHECK(hipMalloc(&Out32, N * sizeof(unsigned)));
  CHECK(hipMalloc(&In64, N * sizeof(unsigned long long)));
  CHECK(hipMalloc(&Out64, N * sizeof(unsigned long long)));
  CHECK(hipMalloc(&In16, BlockSize * sizeof(unsigned short)));
  CHECK(hipMalloc(&Out16, BlockSize * sizeof(unsigned short)));
  CHECK(hipMalloc(&OutLocal, BlockSize * sizeof(unsigned)));
  CHECK(hipMemcpy(In32, HIn32.data(), N * sizeof(unsigned),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(In64, HIn64.data(), N * sizeof(unsigned long long),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(In16, HIn16.data(), BlockSize * sizeof(unsigned short),
                  hipMemcpyHostToDevice));

  volatileAccess<<<NumBlocks, BlockSize>>>(In32, In64, In16, Out32, Out64,
                                           Out16);
  CHECK(hipGetLastError());
  volatileLocal<<<1, BlockSize>>>(OutLocal);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());

  CHECK(hipMemcpy(HOut32.data(), Out32, N * sizeof(unsigned),
                  hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(HOut64.data(), Out64, N * sizeof(unsigned long long),
                  hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(HOut16.data(), Out16, BlockSize * sizeof(unsigned short),
                  hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(HOutLocal.data(), OutLocal, BlockSize * sizeof(unsigned),
                  hipMemcpyDeviceToHost));

  int Errors = 0;
  for (int I = 0; I < N; ++I) {
    if (HOut32[I] != HIn32[I] + 1u) {
      if (Errors++ < 5)
        printf("Out32[%d] = %u, expected %u\n", I, HOut32[I], HIn32[I] + 1u);
    }
    if (HOut64[I] != HIn64[I] + 1ull) {
      if (Errors++ < 5)
        printf("Out64[%d] = %llu, expected %llu\n", I, HOut64[I],
               HIn64[I] + 1ull);
    }
  }
  for (int I = 0; I < BlockSize; ++I) {
    if (HOut16[I] != (unsigned short)(HIn16[I] + 1)) {
      if (Errors++ < 5)
        printf("Out16[%d] = %u, expected %u\n", I, HOut16[I], HIn16[I] + 1);
    }
    if (HOutLocal[I] != (unsigned)((I + 1) % BlockSize)) {
      if (Errors++ < 5)
        printf("OutLocal[%d] = %u, expected %u\n", I, HOutLocal[I],
               (I + 1) % BlockSize);
    }
  }

  Errors += runOnHostVisibleMemory();

  CHECK(hipFree(In32));
  CHECK(hipFree(Out32));
  CHECK(hipFree(In64));
  CHECK(hipFree(Out64));
  CHECK(hipFree(In16));
  CHECK(hipFree(Out16));
  CHECK(hipFree(OutLocal));

  if (Errors) {
    printf("FAILED: %d mismatches\n", Errors);
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
