// Reproduces: the C99 libcalls behind libstdc++'s <cmath> float overloads are
// undefined in the device module, so the driver refuses to link it.
//
// libstdc++ implements std::tgamma(float) and friends as constexpr inline
// wrappers around __builtin_tgammaf. HIP makes constexpr functions implicitly
// __host__ __device__, so in device code they win overload resolution over
// chipStar's extern "C++" __device__ float tgamma(float) unless sp_math.hh
// re-exports that one into namespace std (it does for acosh, lgamma, log1p,
// nearbyint, nextafter, ... but not for these). Builtins with no LLVM
// intrinsic then lower to calls of the C-linkage names:
//
//   asinhf atanhf fdimf ilogbf logbf nexttowardf remainderf scalblnf scalbnf
//   tgammaf
//
// Nothing defines those: sp_math.hh only has C++-linkage overloads, and
// bitcode/c_to_opencl.def, the table that maps the C names of the other
// <cmath> functions (expf, cosf, lgammaf, ...) to OpenCL builtins inside the
// device library hipspv-link pulls in at compile time, lacks them. The SPIR-V
// module imports them and the JIT fails:
//
//   CHIP warning : Missing definition for 'tgammaf'
//   error : unresolved external symbol tgammaf at offset 1644 in instructions
//   segment #4336 (aka kernel : _ZN6Kokkos4ImplL32hip_parallel_launch_...)
//   CHIP warning : Module created with multiple inputs is unlinked.
//   Attempting fallback: create separate modules and link.
//
// Found with Kokkos_CoreUnitTest_HIP (TestMathematicalFunctions.hpp, float
// variants) on Aurora PVC, where it costs a 340 s device compile plus the
// multi-input relink fallback before failing.
//
// The same route exists for long double, which is a 64-bit double on spirv64:
// std::tgamma(long double) compiles and reaches tgammal. The double overloads
// take a different route: libstdc++ gets them from <math.h> as plain host
// functions, so chipStar's __device__ overloads win and no C name is formed.
// The unsuffixed C names (tgamma, asinh, ...) are still reachable through
// __builtin_* directly, or through optimizer-synthesised calls (#880's
// pow(2, i) -> ldexpf), so the Builtin flavour below covers them for float and
// double.
//
// Every argument is loaded from memory so nothing constant-folds away. The
// point is that the module links and every symbol resolves; the values are a
// secondary check with a loose tolerance.

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <type_traits>

// (name, argument): every unary <cmath> function libstdc++ implements for
// float through __builtin_*f (GCC 13 <cmath>), plus the classic ones.
#define UNARY_LIST(X)                                                          \
  X(acos, x) X(acosh, xp1) X(asin, x) X(asinh, x) X(atan, x) X(atanh, x)       \
  X(cbrt, x) X(ceil, y) X(cos, x) X(cosh, x) X(erf, x) X(erfc, x) X(exp, x)     \
  X(exp2, x) X(expm1, x) X(fabs, nx) X(floor, y) X(lgamma, x) X(log, xp1)      \
  X(log10, xp1) X(log1p, x) X(log2, xp1) X(logb, y) X(nearbyint, y)            \
  X(rint, y) X(round, y) X(sin, x) X(sinh, x) X(sqrt, x) X(tan, x) X(tanh, x)  \
  X(tgamma, x) X(trunc, y)

// (name, lhs, rhs)
#define BINARY_LIST(X)                                                         \
  X(atan2, x, y) X(copysign, x, ny) X(fdim, y, x) X(fmax, x, y) X(fmin, x, y)  \
  X(fmod, y, x) X(hypot, x, y) X(nextafter, x, y) X(pow, x, y)                 \
  X(remainder, y, x)

// Functions with other signatures are spelled out in evalAll below, in this
// order: fma, ldexp, scalbn, scalbln, frexp (+ exponent), modf (+ integral
// part), remquo (+ quotient), nexttoward; then the integer-valued ilogb,
// lrint, lround, llrint, llround.
#define COUNT1(NAME, ARG) +1
#define COUNT2(NAME, A, B) +1
constexpr int NumOut = 0 UNARY_LIST(COUNT1) BINARY_LIST(COUNT2) + 9;
constexpr int NumIOut = 7;

// Per-type wrappers so one evaluator can reach both the std:: overloads and
// the __builtin_* forms that name the C functions directly.
#define DEF_BUILTIN1(NAME, ARG)                                                \
  __host__ __device__ inline float bi_##NAME(float a) {                        \
    return __builtin_##NAME##f(a);                                             \
  }                                                                            \
  __host__ __device__ inline double bi_##NAME(double a) {                      \
    return __builtin_##NAME(a);                                                \
  }
#define DEF_BUILTIN2(NAME, A, B)                                               \
  __host__ __device__ inline float bi_##NAME(float a, float b) {               \
    return __builtin_##NAME##f(a, b);                                          \
  }                                                                            \
  __host__ __device__ inline double bi_##NAME(double a, double b) {            \
    return __builtin_##NAME(a, b);                                             \
  }
UNARY_LIST(DEF_BUILTIN1)
BINARY_LIST(DEF_BUILTIN2)
__host__ __device__ inline float bi_fma(float a, float b, float c) {
  return __builtin_fmaf(a, b, c);
}
__host__ __device__ inline double bi_fma(double a, double b, double c) {
  return __builtin_fma(a, b, c);
}
__host__ __device__ inline float bi_ldexp(float a, int n) {
  return __builtin_ldexpf(a, n);
}
__host__ __device__ inline double bi_ldexp(double a, int n) {
  return __builtin_ldexp(a, n);
}
__host__ __device__ inline float bi_scalbn(float a, int n) {
  return __builtin_scalbnf(a, n);
}
__host__ __device__ inline double bi_scalbn(double a, int n) {
  return __builtin_scalbn(a, n);
}
__host__ __device__ inline float bi_scalbln(float a, long n) {
  return __builtin_scalblnf(a, n);
}
__host__ __device__ inline double bi_scalbln(double a, long n) {
  return __builtin_scalbln(a, n);
}
__host__ __device__ inline float bi_nexttoward(float a, long double b) {
  return __builtin_nexttowardf(a, b);
}
__host__ __device__ inline double bi_nexttoward(double a, long double b) {
  return __builtin_nexttoward(a, b);
}
__host__ __device__ inline int bi_ilogb(float a) { return __builtin_ilogbf(a); }
__host__ __device__ inline int bi_ilogb(double a) { return __builtin_ilogb(a); }
__host__ __device__ inline long bi_lrint(float a) { return __builtin_lrintf(a); }
__host__ __device__ inline long bi_lrint(double a) { return __builtin_lrint(a); }
__host__ __device__ inline long bi_lround(float a) {
  return __builtin_lroundf(a);
}
__host__ __device__ inline long bi_lround(double a) {
  return __builtin_lround(a);
}
__host__ __device__ inline long long bi_llrint(float a) {
  return __builtin_llrintf(a);
}
__host__ __device__ inline long long bi_llrint(double a) {
  return __builtin_llrint(a);
}
__host__ __device__ inline long long bi_llround(float a) {
  return __builtin_llroundf(a);
}
__host__ __device__ inline long long bi_llround(double a) {
  return __builtin_llround(a);
}

// One body for the kernels and for the host reference, so the two cannot
// drift apart. T is the argument type, OutT the storage type (long double is
// 8 bytes on the device and 16 on the host, so it is stored as double).
template <typename T, typename OutT, bool UseBuiltin>
__host__ __device__ void evalAll(T x, T y, int n, long ln, OutT *Out,
                                 long long *IOut) {
  constexpr bool IsLongDouble = std::is_same<T, long double>::value;
  constexpr bool IsDouble = std::is_same<T, double>::value;
  T xp1 = x + T(1);
  T nx = -x;
  T ny = -y;
  int i = 0;
  int j = 0;
#define EVAL1(NAME, ARG)                                                       \
  if constexpr (UseBuiltin)                                                    \
    Out[i++] = (OutT)bi_##NAME(ARG);                                           \
  else                                                                         \
    Out[i++] = (OutT)std::NAME(ARG);
#define EVAL2(NAME, A, B)                                                      \
  if constexpr (UseBuiltin)                                                    \
    Out[i++] = (OutT)bi_##NAME(A, B);                                          \
  else                                                                         \
    Out[i++] = (OutT)std::NAME(A, B);
  UNARY_LIST(EVAL1)
  BINARY_LIST(EVAL2)
#undef EVAL1
#undef EVAL2
  if constexpr (UseBuiltin) {
    Out[i++] = (OutT)bi_fma(x, y, x);
    Out[i++] = (OutT)bi_ldexp(x, n);
    Out[i++] = (OutT)bi_scalbn(x, n);
    Out[i++] = (OutT)bi_scalbln(x, ln);
  } else {
    Out[i++] = (OutT)std::fma(x, y, x);
    Out[i++] = (OutT)std::ldexp(x, n);
    Out[i++] = (OutT)std::scalbn(x, n);
    Out[i++] = (OutT)std::scalbln(x, ln);
  }
  // frexp, modf and remquo take a pointer. Through std:: they resolve to
  // chipStar's __device__ overloads on every type that has a viable one, so
  // they never form a C name; they are evaluated through std:: in both
  // flavours. std::modf(long double, long double *) has no device candidate
  // at all, so that slot is a placeholder for long double.
  int e = 0;
  Out[i++] = (OutT)std::frexp(y, &e);
  IOut[j++] = e;
  if constexpr (IsLongDouble) {
    Out[i++] = (OutT)(y - std::trunc(y));
    Out[i++] = (OutT)std::trunc(y);
  } else {
    T ip = 0;
    Out[i++] = (OutT)std::modf(y, &ip);
    Out[i++] = (OutT)ip;
  }
  int q = 0;
  Out[i++] = (OutT)std::remquo(y, x, &q);
  IOut[j++] = q;
  // std::nexttoward(double, long double) is ambiguous in device code (the
  // exact match is libstdc++'s host-only one), so that slot is a placeholder.
  if constexpr (UseBuiltin)
    Out[i++] = (OutT)bi_nexttoward(x, (long double)y);
  else if constexpr (IsDouble)
    Out[i++] = (OutT)x;
  else
    Out[i++] = (OutT)std::nexttoward(x, (long double)y);
  if constexpr (UseBuiltin) {
    IOut[j++] = bi_ilogb(y);
    IOut[j++] = bi_lrint(y);
    IOut[j++] = bi_lround(y);
    IOut[j++] = bi_llrint(y);
    IOut[j++] = bi_llround(y);
  } else {
    IOut[j++] = std::ilogb(y);
    IOut[j++] = std::lrint(y);
    IOut[j++] = std::lround(y);
    IOut[j++] = std::llrint(y);
    IOut[j++] = std::llround(y);
  }
}

template <typename T, typename OutT, bool UseBuiltin>
__global__ void evalKernel(const OutT *In, const int *IIn, OutT *Out,
                           long long *IOut) {
  evalAll<T, OutT, UseBuiltin>((T)In[0], (T)In[1], IIn[0], (long)IIn[1], Out,
                               IOut);
}

#define NAME1(NAME, ARG) #NAME,
#define NAME2(NAME, A, B) #NAME,
static const char *OutNames[NumOut] = {
    UNARY_LIST(NAME1) BINARY_LIST(NAME2) "fma",  "ldexp",  "scalbn",
    "scalbln",        "frexp",           "modf", "modf.ip", "remquo",
    "nexttoward"};
static const char *IOutNames[NumIOut] = {"frexp.exp", "remquo.quo", "ilogb",
                                         "lrint",     "lround",     "llrint",
                                         "llround"};

template <typename T, typename OutT, bool UseBuiltin>
int runOne(const char *Label, double Tol) {
  const OutT HostIn[2] = {(OutT)0.75, (OutT)2.5};
  const int HostIIn[2] = {3, 3};
  OutT *In, *Out;
  int *IIn;
  long long *IOut;
  if (hipMalloc(&In, sizeof(HostIn)) != hipSuccess ||
      hipMalloc(&IIn, sizeof(HostIIn)) != hipSuccess ||
      hipMalloc(&Out, NumOut * sizeof(OutT)) != hipSuccess ||
      hipMalloc(&IOut, NumIOut * sizeof(long long)) != hipSuccess) {
    printf("FAIL: %s: hipMalloc\n", Label);
    return 1;
  }
  (void)hipMemcpy(In, HostIn, sizeof(HostIn), hipMemcpyHostToDevice);
  (void)hipMemcpy(IIn, HostIIn, sizeof(HostIIn), hipMemcpyHostToDevice);

  evalKernel<T, OutT, UseBuiltin><<<1, 1>>>(In, IIn, Out, IOut);
  hipError_t LaunchErr = hipGetLastError();
  hipError_t SyncErr = hipDeviceSynchronize();
  if (LaunchErr != hipSuccess || SyncErr != hipSuccess) {
    printf("FAIL: %s: launch=%s sync=%s\n", Label, hipGetErrorString(LaunchErr),
           hipGetErrorString(SyncErr));
    return 1;
  }

  OutT HostOut[NumOut];
  long long HostIOut[NumIOut];
  (void)hipMemcpy(HostOut, Out, sizeof(HostOut), hipMemcpyDeviceToHost);
  (void)hipMemcpy(HostIOut, IOut, sizeof(HostIOut), hipMemcpyDeviceToHost);
  (void)hipFree(In);
  (void)hipFree(IIn);
  (void)hipFree(Out);
  (void)hipFree(IOut);

  OutT Expected[NumOut];
  long long IExpected[NumIOut];
  evalAll<T, OutT, UseBuiltin>((T)HostIn[0], (T)HostIn[1], HostIIn[0],
                               (long)HostIIn[1], Expected, IExpected);

  int Mismatches = 0;
  for (int K = 0; K < NumOut; ++K) {
    double Got = (double)HostOut[K], Want = (double)Expected[K];
    double Scale = std::fabs(Want) > 1.0 ? std::fabs(Want) : 1.0;
    if (!(std::fabs(Got - Want) <= Tol * Scale)) {
      printf("MISMATCH: %s: %s: got %.9g expected %.9g\n", Label, OutNames[K],
             Got, Want);
      ++Mismatches;
    }
  }
  for (int K = 0; K < NumIOut; ++K)
    if (HostIOut[K] != IExpected[K]) {
      printf("MISMATCH: %s: %s: got %lld expected %lld\n", Label, IOutNames[K],
             HostIOut[K], IExpected[K]);
      ++Mismatches;
    }
  if (Mismatches) {
    printf("FAIL: %s: %d mismatches\n", Label, Mismatches);
    return 1;
  }
  printf("%s: ok (%d values, %d integers)\n", Label, NumOut, NumIOut);
  return 0;
}

int main() {
  int Failures = 0;
  Failures += runOne<float, float, false>("float via std::", 1e-4);
  Failures += runOne<float, float, true>("float via __builtin_*f", 1e-4);
  Failures += runOne<double, double, false>("double via std::", 1e-9);
  Failures += runOne<double, double, true>("double via __builtin_*", 1e-9);
  Failures += runOne<long double, double, false>("long double via std::", 1e-9);
  if (Failures) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}
