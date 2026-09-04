// Reproduces CHIP-SPV/chipStar#1586: devicelib declares both an api_half
// (_Float16) and a double overload of these math functions at global scope,
// and int -> _Float16 and int -> double are floating-integral conversions of
// the same rank. Neither candidate is better, so a device-side call with an
// integer argument is ambiguous. [c.math.fpret] requires such a call to be
// treated as double.
//
// The unqualified form is ambiguous with any standard library. The std::
// form is additionally ambiguous under libc++, whose integral <cmath>
// overloads are not available in the device pass; libstdc++ hides that one
// because its integral overloads are constexpr, hence implicitly
// __host__ __device__.
//
// Compile-only test: it never has to run, it only has to compile.
#include <hip/hip_runtime.h>
#include <cmath>

#define CHECK_INT_ARG(NAME)                                                    \
  __device__ double unqualified_##NAME(int Arg) { return NAME(Arg); }          \
  __device__ double qualified_##NAME(int Arg) { return std::NAME(Arg); }

CHECK_INT_ARG(ceil)
CHECK_INT_ARG(cos)
CHECK_INT_ARG(exp)
CHECK_INT_ARG(floor)
CHECK_INT_ARG(log)
CHECK_INT_ARG(log10)
CHECK_INT_ARG(log2)
CHECK_INT_ARG(sin)
CHECK_INT_ARG(sqrt)
CHECK_INT_ARG(trunc)

// rint is declared for api_half and double too, but is not hoisted into
// namespace std by devicelib, so only the unqualified form is checked.
__device__ double unqualified_rint(int Arg) { return rint(Arg); }

// Other integer types must resolve the same way.
__device__ double sqrtShort(short Arg) { return sqrt(Arg); }
__device__ double sqrtLong(long Arg) { return sqrt(Arg); }
__device__ double sqrtUnsigned(unsigned Arg) { return sqrt(Arg); }
__device__ double sqrtChar(char Arg) { return sqrt(Arg); }
__device__ double sqrtBool(bool Arg) { return sqrt(Arg); }

// The floating-point overloads must keep working, and an int argument must
// pick double, not _Float16: sqrt(16) is 4.0 exactly either way, but
// exp(11) overflows _Float16 (max 65504) and is representable as double.
__device__ double expIntNotHalf(int Arg) { return exp(Arg); }
__device__ _Float16 sqrtHalf(_Float16 Arg) { return sqrt(Arg); }
__device__ float sqrtFloat(float Arg) { return sqrt(Arg); }
__device__ double sqrtDouble(double Arg) { return sqrt(Arg); }

int main() { return 0; }
