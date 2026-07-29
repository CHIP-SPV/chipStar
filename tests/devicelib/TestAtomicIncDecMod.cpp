// Reproduces: __builtin_amdgcn_atomic_inc32 / __builtin_amdgcn_atomic_dec32 are
// undeclared on chipStar's SPIR-V target. desul (bundled with Kokkos) uses them
// to implement atomic_fetch_inc_mod / atomic_fetch_dec_mod, so every HIP
// translation unit that pulls in desul/atomics/Fetch_Op_HIP.hpp fails to
// compile.
//
// inc32 semantics: old = *ptr; *ptr = (old >= val) ? 0 : old + 1; return old;
// dec32 semantics: old = *ptr; *ptr = (old == 0 || old > val) ? val : old - 1;
//                  return old;

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

// One thread walks the counter through a full wrap cycle and records every
// value returned, so the exact modular semantics are checked and not just the
// endpoint.
__global__ void incSequence(unsigned *Counter, unsigned *Out, unsigned Mod,
                            unsigned Steps) {
  for (unsigned I = 0; I < Steps; ++I)
    Out[I] = __builtin_amdgcn_atomic_inc32(Counter, Mod, __ATOMIC_RELAXED,
                                           "agent");
}

__global__ void decSequence(unsigned *Counter, unsigned *Out, unsigned Mod,
                            unsigned Steps) {
  for (unsigned I = 0; I < Steps; ++I)
    Out[I] = __builtin_amdgcn_atomic_dec32(Counter, Mod, __ATOMIC_RELAXED,
                                           "agent");
}

// Every thread bumps the same wrapping counter once; the final value must be
// NumThreads modulo (Mod + 1).
__global__ void incContended(unsigned *Counter, unsigned Mod) {
  __builtin_amdgcn_atomic_inc32(Counter, Mod, __ATOMIC_RELAXED, "agent");
}

int main() {
  constexpr unsigned Mod = 4;   // counter wraps back to 0 after reaching Mod
  constexpr unsigned Steps = 12;

  unsigned *Counter, *Out;
  CHECK(hipMalloc(&Counter, sizeof(unsigned)));
  CHECK(hipMalloc(&Out, Steps * sizeof(unsigned)));

  unsigned Host[Steps];
  int Errors = 0;

  // --- inc ---
  unsigned Zero = 0;
  CHECK(hipMemcpy(Counter, &Zero, sizeof(unsigned), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(incSequence, dim3(1), dim3(1), 0, 0, Counter, Out, Mod,
                     Steps);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(Host, Out, sizeof(Host), hipMemcpyDeviceToHost));
  for (unsigned I = 0; I < Steps; ++I) {
    unsigned Expected = I % (Mod + 1);
    if (Host[I] != Expected) {
      printf("inc32 step %u: got %u expected %u\n", I, Host[I], Expected);
      ++Errors;
    }
  }

  // --- dec ---
  unsigned Start = Mod;
  CHECK(hipMemcpy(Counter, &Start, sizeof(unsigned), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(decSequence, dim3(1), dim3(1), 0, 0, Counter, Out, Mod,
                     Steps);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(Host, Out, sizeof(Host), hipMemcpyDeviceToHost));
  for (unsigned I = 0; I < Steps; ++I) {
    unsigned Expected = Mod - (I % (Mod + 1));
    if (Host[I] != Expected) {
      printf("dec32 step %u: got %u expected %u\n", I, Host[I], Expected);
      ++Errors;
    }
  }

  // --- contended inc ---
  constexpr unsigned NumThreads = 1024;
  CHECK(hipMemcpy(Counter, &Zero, sizeof(unsigned), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(incContended, dim3(4), dim3(NumThreads / 4), 0, 0, Counter,
                     Mod);
  CHECK(hipDeviceSynchronize());
  unsigned Final = 0;
  CHECK(hipMemcpy(&Final, Counter, sizeof(unsigned), hipMemcpyDeviceToHost));
  unsigned ExpectedFinal = NumThreads % (Mod + 1);
  if (Final != ExpectedFinal) {
    printf("contended inc32: got %u expected %u\n", Final, ExpectedFinal);
    ++Errors;
  }

  CHECK(hipFree(Counter));
  CHECK(hipFree(Out));

  if (Errors) {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
