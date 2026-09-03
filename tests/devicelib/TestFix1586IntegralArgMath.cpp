// Reproduces CHIP-SPV/chipStar#1586: devicelib declares float, double and
// __half overloads of the math functions but none taking an integer, so an
// integer argument converts equally well to more than one of them and the
// call is ambiguous. C++ requires <cmath> to provide "sufficient additional
// overloads" so that integer arguments behave as if converted to double.
// Compile-only test: it never has to run, it only has to compile.
#include <hip/hip_runtime.h>
#include <cmath>

#define CHECK_UNARY(NAME)                                                      \
  __device__ double check_##NAME(int A) { return NAME(A); }
#define CHECK_BINARY(NAME)                                                     \
  __device__ double checkII_##NAME(int A, int B) { return NAME(A, B); }        \
  __device__ double checkDI_##NAME(double A, int B) { return NAME(A, B); }

CHECK_UNARY(acosh)
CHECK_UNARY(asinh)
CHECK_UNARY(atanh)
CHECK_UNARY(ceil)
CHECK_UNARY(cos)
CHECK_UNARY(cospi)
CHECK_UNARY(erf)
CHECK_UNARY(exp)
CHECK_UNARY(exp10)
CHECK_UNARY(exp2)
CHECK_UNARY(floor)
CHECK_UNARY(lgamma)
CHECK_UNARY(log)
CHECK_UNARY(log10)
CHECK_UNARY(log1p)
CHECK_UNARY(log2)
CHECK_UNARY(logb)
CHECK_UNARY(nearbyint)
CHECK_UNARY(rint)
CHECK_UNARY(rsqrt)
CHECK_UNARY(sin)
CHECK_UNARY(sinpi)
CHECK_UNARY(sqrt)
CHECK_UNARY(tan)
CHECK_UNARY(tgamma)
CHECK_UNARY(trunc)

CHECK_BINARY(atan2)
CHECK_BINARY(copysign)
CHECK_BINARY(fdim)
CHECK_BINARY(fmin)
CHECK_BINARY(nextafter)
CHECK_BINARY(remainder)

// The same calls through namespace std, which is how portable code spells
// them and how the boost and hip-tests failures showed up.
__device__ double stdSqrtInt(int A) { return std::sqrt(A); }
__device__ double stdPowIntInt(int A, int B) { return std::pow(A, B); }

// Narrow integer types too: __hip_numeric_limits had no specialization for
// them, so the integral overload never fired.
__device__ double stdSqrtShort(short A) { return std::sqrt(A); }
__device__ double stdSqrtChar(char A) { return std::sqrt(A); }
__device__ double stdSqrtBool(bool A) { return std::sqrt(A); }

__global__ void k(double *Out) { *Out = check_sqrt(4) + stdSqrtInt(4); }

int main() { return 0; }
