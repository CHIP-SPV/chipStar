// Reproduces CHIP-SPV/chipStar#1497: 8 and 16 bit atomics reach the SPIR-V
// module as OpAtomicLoad / OpAtomicStore / OpAtomicCompareExchange and friends
// on OpTypeInt 8 and 16, which OpenCL SPIR-V consumers do not implement (IGC
// fails the module build with undefined references to
// __spirv_AtomicLoad(char AS4*, int, int) and friends).
//
// Every kernel below hits all four byte lanes (or both short lanes) of a 32 bit
// word from different threads at the same time, so a lowering that operates on
// the containing word without a compare-and-swap loop is caught as lost updates
// or clobbered neighbour lanes, not just a build failure.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr int NumBlocks = 8;
static constexpr int BlockSize = 256;
static constexpr int NumThreads = NumBlocks * BlockSize; // 2048
static constexpr int NumElems = 64; // 32 threads share every element
static constexpr int ThreadsPerElem = NumThreads / NumElems;
// storeLoadKernel tags are Elem * Ranks + Rank + 1 and must stay non-zero and
// inside 8 bits, so it runs in one block of StoreLoadThreads threads.
static constexpr int StoreLoadThreads = 128;

#define CHECK(Cmd)                                                             \
  do {                                                                         \
    hipError_t Err = (Cmd);                                                    \
    if (Err != hipSuccess) {                                                   \
      printf("FAILED: %s at %s:%d\n", hipGetErrorString(Err), __FILE__,        \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

template <typename T>
__global__ void fetchAddKernel(T *Data, int Iters) {
  int Tid = blockIdx.x * blockDim.x + threadIdx.x;
  T *Slot = Data + (Tid % NumElems);
  for (int I = 0; I < Iters; ++I)
    __hip_atomic_fetch_add(Slot, T(1), __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
__global__ void fetchSubKernel(T *Data, int Iters) {
  int Tid = blockIdx.x * blockDim.x + threadIdx.x;
  T *Slot = Data + (Tid % NumElems);
  for (int I = 0; I < Iters; ++I)
    __hip_atomic_fetch_sub(Slot, T(1), __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
}

// Each thread ORs in / ANDs out one bit of its lane, then max / min its rank.
template <typename T>
__global__ void bitwiseMinMaxKernel(T *OrData, T *AndData, T *MaxData,
                                    T *MinData) {
  int Tid = blockIdx.x * blockDim.x + threadIdx.x;
  int Elem = Tid % NumElems;
  int Rank = Tid / NumElems; // 0 .. ThreadsPerElem - 1
  T Bit = T(1) << (Rank % 8);
  __hip_atomic_fetch_or(OrData + Elem, Bit, __ATOMIC_RELAXED,
                        __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_fetch_and(AndData + Elem, T(~Bit), __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_fetch_max(MaxData + Elem, T(Rank + 1), __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_fetch_min(MinData + Elem, T(Rank + 1), __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
}

// Each thread swaps in its rank + 1 and records what it swapped out.
template <typename T>
__global__ void exchangeKernel(T *Data, T *Old) {
  int Tid = blockIdx.x * blockDim.x + threadIdx.x;
  Old[Tid] = __hip_atomic_exchange(Data + (Tid % NumElems),
                                   T(Tid / NumElems + 1), __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
}

// Increment through a CAS loop, then one CAS that must fail and leave the
// value alone while reporting the current one.
template <typename T>
__global__ void casKernel(T *Data, int Iters, int *Errors) {
  int Tid = blockIdx.x * blockDim.x + threadIdx.x;
  T *Slot = Data + (Tid % NumElems);
  for (int I = 0; I < Iters; ++I) {
    T Expected = __hip_atomic_load(Slot, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
    while (!__hip_atomic_compare_exchange_strong(
        Slot, &Expected, T(Expected + 1), __ATOMIC_RELAXED, __ATOMIC_RELAXED,
        __HIP_MEMORY_SCOPE_AGENT)) {
    }
  }
  T Bogus = T(-1);
  T Seen = Bogus;
  if (__hip_atomic_compare_exchange_strong(Slot, &Seen, T(0), __ATOMIC_RELAXED,
                                           __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT))
    atomicAdd(Errors, 1);
  if (Seen == Bogus)
    atomicAdd(Errors, 1);
}

// One block: every thread stores a tag unique to its (element, rank) pair,
// then every thread loads its element back and checks the tag belongs to that
// element. A store that rewrites the neighbour lanes non-atomically leaks a
// foreign tag into them.
template <typename T>
__global__ void storeLoadKernel(T *Data, int *Errors) {
  int Tid = threadIdx.x;
  int Elem = Tid % NumElems;
  int Rank = Tid / NumElems;
  int Ranks = blockDim.x / NumElems;
  __hip_atomic_store(Data + Elem, T(Elem * Ranks + Rank + 1), __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_AGENT);
  __syncthreads();
  T V = __hip_atomic_load(Data + Elem, __ATOMIC_ACQUIRE,
                          __HIP_MEMORY_SCOPE_AGENT);
  int Tag = int(V) - 1;
  if (Tag < Elem * Ranks || Tag >= (Elem + 1) * Ranks)
    atomicAdd(Errors, 1);
}

static int Failures = 0;

template <typename T> static T *toDevice(const std::vector<T> &H) {
  T *D;
  CHECK(hipMalloc(&D, H.size() * sizeof(T)));
  CHECK(hipMemcpy(D, H.data(), H.size() * sizeof(T), hipMemcpyHostToDevice));
  return D;
}

template <typename T> static std::vector<T> toHost(T *D, size_t N) {
  std::vector<T> H(N);
  CHECK(hipMemcpy(H.data(), D, N * sizeof(T), hipMemcpyDeviceToHost));
  CHECK(hipFree(D));
  return H;
}

template <typename T>
static void expectAll(const char *What, const std::vector<T> &Got, T Want) {
  for (size_t I = 0; I < Got.size(); ++I)
    if (Got[I] != Want) {
      printf("FAILED: %s[%zu] = %d, expected %d\n", What, I, int(Got[I]),
             int(Want));
      ++Failures;
      return;
    }
}

template <typename T> static void testFetchAddSub(const char *Name, int Iters) {
  T *Add = toDevice(std::vector<T>(NumElems, T(0)));
  T *Sub = toDevice(std::vector<T>(NumElems, T(0)));
  fetchAddKernel<T><<<NumBlocks, BlockSize>>>(Add, Iters);
  fetchSubKernel<T><<<NumBlocks, BlockSize>>>(Sub, Iters);
  CHECK(hipDeviceSynchronize());
  expectAll(Name, toHost(Add, NumElems), T(ThreadsPerElem * Iters));
  expectAll(Name, toHost(Sub, NumElems), T(-(ThreadsPerElem * Iters)));
}

template <typename T> static void testBitwiseMinMax(const char *Name) {
  T *Or = toDevice(std::vector<T>(NumElems, T(0)));
  T *And = toDevice(std::vector<T>(NumElems, T(-1)));
  T *Max = toDevice(std::vector<T>(NumElems, T(0)));
  T *Min = toDevice(std::vector<T>(NumElems, T(ThreadsPerElem + 1)));
  bitwiseMinMaxKernel<T><<<NumBlocks, BlockSize>>>(Or, And, Max, Min);
  CHECK(hipDeviceSynchronize());
  expectAll(Name, toHost(Or, NumElems), T(0xFF));
  expectAll(Name, toHost(And, NumElems), T(~0xFF));
  expectAll(Name, toHost(Max, NumElems), T(ThreadsPerElem));
  expectAll(Name, toHost(Min, NumElems), T(1));
}

template <typename T> static void testExchange(const char *Name) {
  T *Data = toDevice(std::vector<T>(NumElems, T(0)));
  T *Old = toDevice(std::vector<T>(NumThreads, T(0)));
  exchangeKernel<T><<<NumBlocks, BlockSize>>>(Data, Old);
  CHECK(hipDeviceSynchronize());
  std::vector<T> Final = toHost(Data, NumElems);
  std::vector<T> Olds = toHost(Old, NumThreads);
  // Per element, {swapped-out values} + {final value} must be exactly
  // {0, 1, ..., ThreadsPerElem}: nothing lost, nothing duplicated.
  for (int E = 0; E < NumElems; ++E) {
    std::vector<int> Seen;
    for (int R = 0; R < ThreadsPerElem; ++R)
      Seen.push_back(int(Olds[R * NumElems + E]));
    Seen.push_back(int(Final[E]));
    std::sort(Seen.begin(), Seen.end());
    for (int I = 0; I <= ThreadsPerElem; ++I)
      if (Seen[I] != I) {
        printf("FAILED: %s exchange multiset broken at element %d\n", Name, E);
        ++Failures;
        return;
      }
  }
}

template <typename T> static void testCas(const char *Name, int Iters) {
  T *Data = toDevice(std::vector<T>(NumElems, T(0)));
  int *Errors = toDevice(std::vector<int>(1, 0));
  casKernel<T><<<NumBlocks, BlockSize>>>(Data, Iters, Errors);
  CHECK(hipDeviceSynchronize());
  expectAll(Name, toHost(Data, NumElems), T(ThreadsPerElem * Iters));
  int Errs = toHost(Errors, 1)[0];
  if (Errs) {
    printf("FAILED: %s failing CAS misbehaved %d times\n", Name, Errs);
    ++Failures;
  }
}

template <typename T> static void testStoreLoad(const char *Name) {
  T *Data = toDevice(std::vector<T>(NumElems, T(0)));
  int *Errors = toDevice(std::vector<int>(1, 0));
  storeLoadKernel<T><<<1, StoreLoadThreads>>>(Data, Errors);
  CHECK(hipDeviceSynchronize());
  std::vector<T> Final = toHost(Data, NumElems);
  int Ranks = StoreLoadThreads / NumElems;
  for (int E = 0; E < NumElems; ++E) {
    int Tag = int(Final[E]) - 1;
    if (Tag < E * Ranks || Tag >= (E + 1) * Ranks) {
      printf("FAILED: %s element %d holds foreign tag %d\n", Name, E, Tag);
      ++Failures;
      break;
    }
  }
  int Errs = toHost(Errors, 1)[0];
  if (Errs) {
    printf("FAILED: %s device-side load saw %d foreign tags\n", Name, Errs);
    ++Failures;
  }
}

int main() {
  // Iteration counts keep every count inside the type's range.
  testFetchAddSub<unsigned char>("uchar", 7);   // 32 * 7 = 224
  testFetchAddSub<signed char>("schar", 3);     // 32 * 3 = 96
  testFetchAddSub<unsigned short>("ushort", 1000);
  testFetchAddSub<short>("short", 500);
  testBitwiseMinMax<unsigned char>("uchar");
  testBitwiseMinMax<unsigned short>("ushort");
  testExchange<unsigned char>("uchar");
  testExchange<short>("short");
  testCas<unsigned char>("uchar", 5);           // 32 * 5 = 160
  testCas<signed char>("schar", 3);
  testCas<unsigned short>("ushort", 200);
  testStoreLoad<unsigned char>("uchar");        // tags 1 .. 128
  testStoreLoad<short>("short");

  if (Failures) {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
