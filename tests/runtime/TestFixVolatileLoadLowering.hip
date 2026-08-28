// Reproducer: volatile loads and stores of 32 and 64 bit values in global
// memory reach the SPIR-V producer as plain `load volatile` / `store volatile`
// and end up as OpLoad / OpStore with the Volatile memory operand, which IGC
// treats as ordinary cached accesses. CUDA compiles the same source to
// ld.volatile / st.volatile, which the PTX ISA (8.4.2 "volatile Operation")
// defines as relaxed memory operations at system scope, and code written
// against that (Kokkos::volatile_load in the UnorderedMap insert list walk)
// reads stale L1 data on PVC. chipStar must lower them to relaxed
// device-scope atomic loads and stores, which the SPIR-V producers emit as
// OpAtomicLoad / OpAtomicStore.
//
// This executable checks that the accesses still compute the right values.
// TestFixVolatileLoadLoweringSPIRV.bash compiles this file with --save-temps and
// inspects the lowered device bitcode (and the SPIR-V module) for the atomic
// forms; that is the check that fails without the lowering.

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

// The accesses the lowering must rewrite: 32 and 64 bit volatile loads and
// stores through global memory pointers.
__global__ void volatileAccess(const unsigned *In32,
                               const unsigned long long *In64, unsigned *Out32,
                               unsigned long long *Out64) {
  int Tid = blockIdx.x * blockDim.x + threadIdx.x;
  const volatile unsigned *VIn32 = In32;
  const volatile unsigned long long *VIn64 = In64;
  volatile unsigned *VOut32 = Out32;
  volatile unsigned long long *VOut64 = Out64;
  unsigned A = VIn32[Tid];
  unsigned long long B = VIn64[Tid];
  VOut32[Tid] = A + 1u;
  VOut64[Tid] = B + 1ull;
}

// Accesses the lowering must leave as they are: 16 bit values (OpenCL SPIR-V
// consumers implement 32 and 64 bit atomics only) and work-group local memory
// (never shared across cores, so no L1 staleness to defeat).
__global__ void volatileLeftAlone(const unsigned short *In16,
                                  unsigned short *Out16, unsigned *OutLocal) {
  __shared__ unsigned Scratch[BlockSize];
  int Tid = threadIdx.x;
  const volatile unsigned short *VIn16 = In16;
  volatile unsigned short *VOut16 = Out16;
  volatile unsigned *VScratch = Scratch;
  VOut16[Tid] = VIn16[Tid] + 1;
  VScratch[Tid] = Tid;
  __syncthreads();
  OutLocal[Tid] = VScratch[(Tid + 1) % BlockSize];
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

  volatileAccess<<<NumBlocks, BlockSize>>>(In32, In64, Out32, Out64);
  CHECK(hipGetLastError());
  volatileLeftAlone<<<1, BlockSize>>>(In16, Out16, OutLocal);
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
