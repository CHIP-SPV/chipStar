// Reproduces the Kokkos::UnorderedMap::insert claim-then-link idiom that
// miscounts on Aurora PVC (Kokkos_ContainersUnitTest_HIP hip.UnorderedMap_insert
// reports 901 distinct keys for 900). Mirrors containers/src/Kokkos_UnorderedMap.hpp
// insert() and containers/src/Kokkos_Bitset.hpp (Kokkos develop 93b924764):
//
//   claim  : Bitset::set(i), a relaxed device-scope atomic fetch_or
//   publish: keys[i] = key; memory_fence() (= __threadfence()); relaxed
//            device-scope CAS of the slot index into the hash list head / link
//   read   : list walk through Kokkos::volatile_load of the link and of the key
//   undo   : Bitset::reset(i), a relaxed device-scope atomic fetch_and
//
// A losing inserter that reads a stale key from the winner's slot appends its
// own slot behind it, and the same key ends up twice in one list. Every run
// walks the lists on the host and fails on a duplicate key, an orphaned slot
// (bit set, never linked) or a bad count. The volatile loads are what the
// lowering in HipLowerVolatileAccessesPass turns into relaxed device-scope
// atomic loads; on the CPU OpenCL runtime this passes before and after.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define INVALID 0xFFFFFFFFu

#define CHECK(Cmd)                                                             \
  do {                                                                         \
    hipError_t Err = (Cmd);                                                    \
    if (Err != hipSuccess) {                                                   \
      printf("FAILED: %s at %s:%d\n", hipGetErrorString(Err), __FILE__,        \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static constexpr unsigned Reps = 40;
static constexpr unsigned NumInserts = 90000;
static constexpr unsigned NumKeys = 900;
static constexpr unsigned Dups = NumInserts / NumKeys;
static constexpr unsigned NumNodes = 100000;
static constexpr unsigned BlockSize = 256;

enum { S_SUCCESS = 0, S_EXISTING = 1, S_FAILED = 2, S_FREE_FAIL = 3, S_COUNT = 4 };

struct Table {
  unsigned *Bits;  // Capacity / 32 words
  unsigned *Heads; // NumLists
  unsigned *Next;  // Capacity
  unsigned *Keys;  // Capacity
  unsigned *Stats; // S_COUNT
};

// Kokkos::pod_hash<uint32_t> = MurmurHash3_x86_32(&key, 4, seed 0)
__host__ __device__ inline uint32_t rotl32(uint32_t X, int R) {
  return (X << R) | (X >> (32 - R));
}
__host__ __device__ inline uint32_t murmur3U32(uint32_t Key, uint32_t Seed) {
  uint32_t H = Seed;
  uint32_t K = Key * 0xcc9e2d51u;
  K = rotl32(K, 15) * 0x1b873593u;
  H ^= K;
  H = rotl32(H, 13) * 5u + 0xe6546b64u;
  H ^= 4u;
  H ^= H >> 16;
  H *= 0x85ebca6bu;
  H ^= H >> 13;
  H *= 0xc2b2ae35u;
  H ^= H >> 16;
  return H;
}

// Kokkos::UnorderedMap sizing (Kokkos_UnorderedMap.hpp, Kokkos_UnorderedMap_impl.cpp)
static uint32_t calculateCapacity(uint32_t Hint) {
  return ((static_cast<uint32_t>(7ull * Hint / 6u) + 127u) / 128u) * 128u;
}
static uint32_t findHashSize(uint32_t Size) {
  static const uint32_t Primes[] = {
      85703,  90749,  95783,  100823, 105871, 110909, 115963, 120997,
      126031, 141157, 151237, 161323, 171401, 181499, 191579, 201653};
  for (uint32_t P : Primes)
    if (Size <= P)
      return P;
  return Primes[sizeof(Primes) / sizeof(Primes[0]) - 1];
}

// Kokkos::volatile_load
__device__ __forceinline__ unsigned vload(unsigned *P) {
  return *reinterpret_cast<volatile unsigned *>(P);
}

// Kokkos::Bitset::set: true if this thread flipped the bit 0 -> 1
__device__ __forceinline__ bool bitSet(unsigned *Bits, unsigned I) {
  unsigned Mask = 1u << (I & 31);
  return !(__hip_atomic_fetch_or(&Bits[I >> 5], Mask, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT) &
           Mask);
}
// Kokkos::Bitset::reset: true if the bit was set
__device__ __forceinline__ bool bitReset(unsigned *Bits, unsigned I) {
  unsigned Mask = 1u << (I & 31);
  return __hip_atomic_fetch_and(&Bits[I >> 5], ~Mask, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT) &
         Mask;
}
// Kokkos::Bitset::find_any_unset_near, forward scan, forward hint move.
__device__ __forceinline__ bool findAnyUnsetNear(unsigned *Bits,
                                                 unsigned NumBlocks,
                                                 unsigned LastBlockMask,
                                                 unsigned *Hint) {
  unsigned BlockIdx = *Hint >> 5;
  unsigned Offset = *Hint & 31;
  unsigned Block = __hip_atomic_load(&Bits[BlockIdx], __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT);
  Block = (!LastBlockMask || BlockIdx < NumBlocks - 1) ? ~Block
                                                       : (~Block & LastBlockMask);
  if (Block == 0u) {
    unsigned NB = BlockIdx + 1;
    if (NB >= NumBlocks)
      NB = 0;
    *Hint = NB * 32u + Offset;
    return false;
  }
  unsigned Rot = Offset ? ((Block >> Offset) | (Block << (32 - Offset))) : Block;
  *Hint = (BlockIdx << 5) + ((__builtin_ctz(Rot) + Offset) & 31);
  return true;
}

// Kokkos::UnorderedMap::insert, unbounded, key == value, NoOp insert op.
__device__ void mapInsert(Table T, unsigned Capacity, unsigned NumLists,
                          unsigned Key) {
  const unsigned NumBlocks = (Capacity + 31) / 32;
  const unsigned LastBlockMask =
      (Capacity & 31) ? ((1u << (Capacity & 31)) - 1u) : 0u;
  const unsigned HashList = murmur3U32(Key, 0) % NumLists;
  unsigned *CurrPtr = &T.Heads[HashList];
  unsigned NewIndex = INVALID;
  unsigned IndexHint = static_cast<unsigned>(
      (static_cast<double>(HashList) * Capacity) / NumLists);
  unsigned FindAttempts = 0;

  bool NotDone = true;
  while (NotDone) {
    // Need volatile_load as other threads may be appending.
    unsigned Curr = vload(CurrPtr);
    while (Curr != INVALID && vload(&T.Keys[Curr]) != Key) {
      IndexHint = Curr;
      CurrPtr = &T.Next[Curr];
      Curr = vload(CurrPtr);
    }

    if (Curr != INVALID) {
      if (NewIndex != INVALID && !bitReset(T.Bits, NewIndex))
        atomicAdd(&T.Stats[S_FREE_FAIL], 1u);
      atomicAdd(&T.Stats[S_EXISTING], 1u);
      NotDone = false;
    } else {
      if (NewIndex == INVALID) {
        bool Found =
            findAnyUnsetNear(T.Bits, NumBlocks, LastBlockMask, &IndexHint);
        if (!Found && ++FindAttempts >= NumBlocks) {
          atomicAdd(&T.Stats[S_FAILED], 1u);
          NotDone = false;
        } else if (bitSet(T.Bits, IndexHint)) {
          NewIndex = IndexHint;
          T.Keys[NewIndex] = Key; // plain store, as in Kokkos
          __threadfence();        // Kokkos::memory_fence()
        }
      }
      unsigned Expected = INVALID;
      if (NewIndex != INVALID) {
        __hip_atomic_compare_exchange_strong(CurrPtr, &Expected, NewIndex,
                                             __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT);
        if (Curr == Expected) {
          atomicAdd(&T.Stats[S_SUCCESS], 1u);
          NotDone = false;
        }
      }
    }
  }
}

__global__ void insertKernel(Table T, unsigned Capacity, unsigned NumLists,
                             int Near) {
  unsigned I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= NumInserts)
    return;
  const unsigned Key = Near ? I / Dups : I % NumKeys;
  mapInsert(T, Capacity, NumLists, Key);
}

int main() {
  const unsigned Capacity = calculateCapacity(NumNodes);
  const unsigned NumLists = findHashSize(Capacity);
  const unsigned NumBlocks = (Capacity + 31) / 32;
  printf("inserts=%u keys=%u capacity=%u hash_lists=%u reps=%u\n", NumInserts,
         NumKeys, Capacity, NumLists, Reps);

  Table T;
  CHECK(hipMalloc(&T.Bits, NumBlocks * sizeof(unsigned)));
  CHECK(hipMalloc(&T.Heads, NumLists * sizeof(unsigned)));
  CHECK(hipMalloc(&T.Next, Capacity * sizeof(unsigned)));
  CHECK(hipMalloc(&T.Keys, Capacity * sizeof(unsigned)));
  CHECK(hipMalloc(&T.Stats, S_COUNT * sizeof(unsigned)));

  std::vector<unsigned> Bits(NumBlocks), Heads(NumLists), Next(Capacity),
      Keys(Capacity), Stats(S_COUNT);
  std::vector<unsigned> FirstSlot(NumKeys);
  std::vector<unsigned char> Reached(Capacity);

  unsigned BadRuns = 0, TotalSuccess = 0, TotalExisting = 0;
  for (unsigned R = 0; R < Reps; ++R) {
    const int Near = R & 1; // the Kokkos test alternates near / not near
    CHECK(hipMemset(T.Bits, 0, NumBlocks * sizeof(unsigned)));
    CHECK(hipMemset(T.Heads, 0xFF, NumLists * sizeof(unsigned)));
    CHECK(hipMemset(T.Next, 0xFF, Capacity * sizeof(unsigned)));
    CHECK(hipMemset(T.Keys, 0, Capacity * sizeof(unsigned)));
    CHECK(hipMemset(T.Stats, 0, S_COUNT * sizeof(unsigned)));
    insertKernel<<<(NumInserts + BlockSize - 1) / BlockSize, BlockSize>>>(
        T, Capacity, NumLists, Near);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(Bits.data(), T.Bits, NumBlocks * sizeof(unsigned),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(Heads.data(), T.Heads, NumLists * sizeof(unsigned),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(Next.data(), T.Next, Capacity * sizeof(unsigned),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(Keys.data(), T.Keys, Capacity * sizeof(unsigned),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(Stats.data(), T.Stats, S_COUNT * sizeof(unsigned),
                    hipMemcpyDeviceToHost));

    // Walk every list: count linked slots, duplicate keys, orphaned bits.
    unsigned SetBits = 0, Linked = 0, DupKeys = 0, Orphans = 0, Broken = 0;
    for (unsigned W = 0; W < NumBlocks; ++W)
      SetBits += __builtin_popcount(Bits[W]);
    std::fill(Reached.begin(), Reached.end(), 0);
    std::fill(FirstSlot.begin(), FirstSlot.end(), INVALID);
    for (unsigned L = 0; L < NumLists; ++L) {
      unsigned Curr = Heads[L];
      unsigned Steps = 0;
      while (Curr != INVALID) {
        if (Curr >= Capacity || Reached[Curr] || ++Steps > Capacity ||
            Keys[Curr] >= NumKeys || murmur3U32(Keys[Curr], 0) % NumLists != L) {
          Broken++;
          break;
        }
        Reached[Curr] = 1;
        Linked++;
        unsigned K = Keys[Curr];
        if (FirstSlot[K] != INVALID) {
          DupKeys++;
          printf("rep %u: duplicate key %u in list %u: slots %u and %u\n", R,
                 K, L, FirstSlot[K], Curr);
        } else {
          FirstSlot[K] = Curr;
        }
        Curr = Next[Curr];
      }
    }
    for (unsigned I = 0; I < Capacity; ++I)
      if (((Bits[I >> 5] >> (I & 31)) & 1u) && !Reached[I])
        Orphans++;
    TotalSuccess += Stats[S_SUCCESS];
    TotalExisting += Stats[S_EXISTING];
    bool Ok = SetBits == NumKeys && Linked == NumKeys && DupKeys == 0 &&
              Orphans == 0 && Broken == 0 && Stats[S_FAILED] == 0 &&
              Stats[S_FREE_FAIL] == 0 &&
              Stats[S_SUCCESS] + Stats[S_EXISTING] == NumInserts;
    if (!Ok) {
      BadRuns++;
      printf("rep %u near=%d: bits=%u linked=%u dup_keys=%u orphans=%u "
             "broken=%u success=%u existing=%u failed=%u free_fail=%u\n",
             R, Near, SetBits, Linked, DupKeys, Orphans, Broken,
             Stats[S_SUCCESS], Stats[S_EXISTING], Stats[S_FAILED],
             Stats[S_FREE_FAIL]);
    }
  }

  CHECK(hipFree(T.Bits));
  CHECK(hipFree(T.Heads));
  CHECK(hipFree(T.Next));
  CHECK(hipFree(T.Keys));
  CHECK(hipFree(T.Stats));

  printf("%u runs: %u new keys, %u existing keys (expected %u and %u)\n", Reps,
         TotalSuccess, TotalExisting, Reps * NumKeys,
         Reps * (NumInserts - NumKeys));
  if (BadRuns) {
    printf("FAILED: %u of %u runs miscounted\n", BadRuns, Reps);
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
