// Tests for atomicInc / atomicDec (CUDA API) and their _system / _block scoped
// variants added when the implementation was rewritten on OpenCL 2.0
// atomic_compare_exchange_strong_explicit.
//
// atomicInc semantics: old >= val ? 0 : old + 1
// atomicDec semantics: (old == 0 || old > val) ? val : old - 1

#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(X)                                                               \
  do {                                                                         \
    hipError_t E = (X);                                                        \
    if (E != hipSuccess) {                                                     \
      printf("FAILED: %s -> %s\n", #X, hipGetErrorString(E));                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// One thread walks the counter through a full wrap cycle with the CUDA API.
__global__ void incSequence(unsigned *Counter, unsigned *Out, unsigned Mod,
                            unsigned Steps) {
  for (unsigned I = 0; I < Steps; ++I)
    Out[I] = atomicInc(Counter, Mod);
}

__global__ void decSequence(unsigned *Counter, unsigned *Out, unsigned Mod,
                            unsigned Steps) {
  for (unsigned I = 0; I < Steps; ++I)
    Out[I] = atomicDec(Counter, Mod);
}

// _system variants: same semantics, different memory scope.
__global__ void incSystemSequence(unsigned *Counter, unsigned *Out,
                                  unsigned Mod, unsigned Steps) {
  for (unsigned I = 0; I < Steps; ++I)
    Out[I] = atomicInc_system(Counter, Mod);
}

__global__ void decSystemSequence(unsigned *Counter, unsigned *Out,
                                  unsigned Mod, unsigned Steps) {
  for (unsigned I = 0; I < Steps; ++I)
    Out[I] = atomicDec_system(Counter, Mod);
}

// _block variant on shared memory: NumThreads threads each bump a shared
// counter once; the final value must be NumThreads modulo (Mod + 1).
__global__ void incBlockContended(unsigned *Out, unsigned Mod) {
  __shared__ unsigned Counter;
  if (threadIdx.x == 0)
    Counter = 0;
  __syncthreads();
  atomicInc_block(&Counter, Mod);
  __syncthreads();
  if (threadIdx.x == 0)
    *Out = Counter;
}

__global__ void decBlockContended(unsigned *Out, unsigned Mod) {
  __shared__ unsigned Counter;
  if (threadIdx.x == 0)
    Counter = Mod;
  __syncthreads();
  atomicDec_block(&Counter, Mod);
  __syncthreads();
  if (threadIdx.x == 0)
    *Out = Counter;
}

int main() {
  constexpr unsigned Mod = 4;
  constexpr unsigned Steps = 12;
  constexpr unsigned NumThreads = 64;

  unsigned *Counter, *Out;
  CHECK(hipMalloc(&Counter, sizeof(unsigned)));
  CHECK(hipMalloc(&Out, Steps * sizeof(unsigned)));

  unsigned Host[Steps];
  int Errors = 0;

  // --- atomicInc: single-thread sequence ---
  unsigned Zero = 0;
  CHECK(hipMemcpy(Counter, &Zero, sizeof(unsigned), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(incSequence, dim3(1), dim3(1), 0, 0, Counter, Out, Mod,
                     Steps);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(Host, Out, sizeof(Host), hipMemcpyDeviceToHost));
  for (unsigned I = 0; I < Steps; ++I) {
    unsigned Expected = I % (Mod + 1);
    if (Host[I] != Expected) {
      printf("atomicInc step %u: got %u expected %u\n", I, Host[I], Expected);
      ++Errors;
    }
  }

  // --- atomicDec: single-thread sequence ---
  unsigned Start = Mod;
  CHECK(hipMemcpy(Counter, &Start, sizeof(unsigned), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(decSequence, dim3(1), dim3(1), 0, 0, Counter, Out, Mod,
                     Steps);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(Host, Out, sizeof(Host), hipMemcpyDeviceToHost));
  for (unsigned I = 0; I < Steps; ++I) {
    unsigned Expected = Mod - (I % (Mod + 1));
    if (Host[I] != Expected) {
      printf("atomicDec step %u: got %u expected %u\n", I, Host[I], Expected);
      ++Errors;
    }
  }

  // --- atomicInc_system: single-thread sequence ---
  CHECK(hipMemcpy(Counter, &Zero, sizeof(unsigned), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(incSystemSequence, dim3(1), dim3(1), 0, 0, Counter, Out,
                     Mod, Steps);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(Host, Out, sizeof(Host), hipMemcpyDeviceToHost));
  for (unsigned I = 0; I < Steps; ++I) {
    unsigned Expected = I % (Mod + 1);
    if (Host[I] != Expected) {
      printf("atomicInc_system step %u: got %u expected %u\n", I, Host[I],
             Expected);
      ++Errors;
    }
  }

  // --- atomicDec_system: single-thread sequence ---
  CHECK(hipMemcpy(Counter, &Start, sizeof(unsigned), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(decSystemSequence, dim3(1), dim3(1), 0, 0, Counter, Out,
                     Mod, Steps);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(Host, Out, sizeof(Host), hipMemcpyDeviceToHost));
  for (unsigned I = 0; I < Steps; ++I) {
    unsigned Expected = Mod - (I % (Mod + 1));
    if (Host[I] != Expected) {
      printf("atomicDec_system step %u: got %u expected %u\n", I, Host[I],
             Expected);
      ++Errors;
    }
  }

  // --- atomicInc_block on shared memory: contended ---
  unsigned Result = 0;
  unsigned *DevResult;
  CHECK(hipMalloc(&DevResult, sizeof(unsigned)));
  hipLaunchKernelGGL(incBlockContended, dim3(1), dim3(NumThreads), 0, 0,
                     DevResult, Mod);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(&Result, DevResult, sizeof(unsigned), hipMemcpyDeviceToHost));
  {
    unsigned Expected = NumThreads % (Mod + 1);
    if (Result != Expected) {
      printf("atomicInc_block contended: got %u expected %u\n", Result,
             Expected);
      ++Errors;
    }
  }

  // --- atomicDec_block on shared memory: contended ---
  hipLaunchKernelGGL(decBlockContended, dim3(1), dim3(NumThreads), 0, 0,
                     DevResult, Mod);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(&Result, DevResult, sizeof(unsigned), hipMemcpyDeviceToHost));
  {
    // NumThreads decrements starting from Mod; wraps at 0 back to Mod.
    unsigned Expected = (Mod - NumThreads % (Mod + 1) + (Mod + 1)) % (Mod + 1);
    if (Result != Expected) {
      printf("atomicDec_block contended: got %u expected %u\n", Result,
             Expected);
      ++Errors;
    }
  }

  CHECK(hipFree(Counter));
  CHECK(hipFree(Out));
  CHECK(hipFree(DevResult));

  if (Errors) {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
