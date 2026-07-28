// Every __half device math function declared by hip_fp16.h must be available
// on the SPIR-V platform, not just on AMD/NVIDIA.
//
// hrcp(__half) used to be wrapped in `#if !defined(__HIP_PLATFORM_SPIRV__)`,
// so it silently vanished on chipStar while its bfloat16 counterpart stayed.
// Kokkos' HIP backend binds both in one macro
// (KOKKOS_HIP_HALF_AND_BHALF_UNARY_FUNCTION_IMPL(rcp, hrcp)), so the __half
// call resolved to the __hip_bfloat16 overload and failed to compile, which
// broke every Kokkos-based build (found while porting Trilinos STK).
//
// Compile-only on purpose: the regression is a declaration being preprocessed
// away, which -fsyntax-only catches on every target. Executing these instead
// probes device fp16 library coverage, which varies: the macOS CI runner's
// PoCL build resolves none of _Z4ceilDh, _Z4sqrtDh, _Z4log2Dh, ... in its
// kernel library, while PoCL 6.0/7.1/2025.01.14 on Linux and Arc B570 all
// evaluate them correctly. That is a device property, not a header property.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void callHalfMathFuncs(__half *Out, __half In) {
  int I = 0;
  Out[I++] = hceil(In);
  Out[I++] = hcos(In);
  Out[I++] = hexp(In);
  Out[I++] = hexp10(In);
  Out[I++] = hexp2(In);
  Out[I++] = hfloor(In);
  Out[I++] = hlog(In);
  Out[I++] = hlog10(In);
  Out[I++] = hlog2(In);
  Out[I++] = hrcp(In); // The one that regressed.
  Out[I++] = hrint(In);
  Out[I++] = hrsqrt(In);
  Out[I++] = hsin(In);
  Out[I++] = hsqrt(In);
  Out[I++] = htrunc(In);
}
