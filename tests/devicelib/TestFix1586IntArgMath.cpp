// Reproduces CHIP-SPV/chipStar#1586: an integer argument ties the api_half and
// double overloads of these device math functions. The unqualified form is
// ambiguous with any standard library; the std:: form only under libc++, whose
// integral <cmath> overloads are host-only (libstdc++'s are constexpr, hence
// implicitly __host__ __device__). Compile-only test.
#include <hip/hip_runtime.h>
#include <cmath>

#define CHECK_INT_ARG(NAME)                                                    \
  __device__ double unqualified_##NAME(int Arg) {                              \
    static_assert(__is_same(decltype(NAME(Arg)), double), "");                 \
    return NAME(Arg);                                                          \
  }                                                                            \
  __device__ double qualified_##NAME(int Arg) {                                \
    static_assert(__is_same(decltype(std::NAME(Arg)), double), "");            \
    return std::NAME(Arg);                                                     \
  }

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
CHECK_INT_ARG(rint)

// Other integer types must resolve the same way.
__device__ void otherIntegerTypes(short S, long L, unsigned U, char C, bool B) {
  static_assert(__is_same(decltype(sqrt(S)), double), "");
  static_assert(__is_same(decltype(sqrt(L)), double), "");
  static_assert(__is_same(decltype(sqrt(U)), double), "");
  static_assert(__is_same(decltype(sqrt(C)), double), "");
  static_assert(__is_same(decltype(sqrt(B)), double), "");
}

// An int must pick double, not _Float16 (exp(12) overflows _Float16), and the
// floating-point overloads must keep working.
__device__ double expIntNotHalf(int Arg) {
  static_assert(__is_same(decltype(exp(Arg)), double), "");
  return exp(Arg);
}
__device__ _Float16 sqrtHalf(_Float16 Arg) { return sqrt(Arg); }
__device__ float sqrtFloat(float Arg) { return sqrt(Arg); }
__device__ double sqrtDouble(double Arg) { return sqrt(Arg); }

int main() { return 0; }
