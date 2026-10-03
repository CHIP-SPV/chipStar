/*
 * Copyright (c) 2023 chipStar developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// Math glue for the native Vulkan device library: the C math names map to
// OpenCL builtins, which libclc provides at link time.

#define OVLD __attribute__((overloadable))

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// See c_to_opencl.def for details; these entries are not in it.
#define DEF_UNARY_FN_MAP(FROM_FN_, TO_FN_, TYPE_)                              \
  TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_ x) { return TO_FN_(x); }
#define DEF_BINARY_FN_MAP(FROM_FN_, TO_FN_, TYPE_)                             \
  TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_ x, TYPE_ y) { return TO_FN_(x, y); }
#define DEF_UNARY_FN_MAP_RET(FROM_FN_, TO_FN_, RET_TYPE_, TYPE_)               \
  RET_TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_ x) { return TO_FN_(x); }
#define DEF_BINARY_FN_MAP_MIXED(FROM_FN_, TO_FN_, TYPE_, TYPE2_)               \
  TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_ x, TYPE2_ y) { return TO_FN_(x, y); }
#define DEF_TERNARY_FN_MAP(FROM_FN_, TO_FN_, TYPE_)                            \
  TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_ x, TYPE_ y, TYPE_ z) {                   \
    return TO_FN_(x, y, z);                                                    \
  }
OVLD static long __chip_vk_lround(float x) { return (long)round(x); }
OVLD static long __chip_vk_lround(double x) { return (long)round(x); }
OVLD static long __chip_vk_lrint(float x) { return (long)rint(x); }
OVLD static long __chip_vk_lrint(double x) { return (long)rint(x); }
#include "vulkan_c_to_opencl.def"

// OCML entry points with no OpenCL builtin counterpart, built on libclc's
// erf/erfc/exp/log/cbrt. erfinv/erfcinv start from Giles' approximation
// ("Approximating the erfinv function", GPU Computing Gems) evaluated on
// w = -log(y(2-y)), then take Newton steps on erfc; the tails below about
// 1e-46 and erfcx above 9 lose accuracy (chipStar#1751).
#define GEN_SPECIALS(T, S)                                                     \
  static T giles_##S(T w) {                                                    \
    T p;                                                                       \
    if (w < (T)6.25) {                                                         \
      w -= (T)3.125;                                                           \
      p = (T)-3.6444120640178196996e-21;                                       \
      p = (T)-1.685059138182016589e-19 + p * w;                                \
      p = (T)1.2858480715256400167e-18 + p * w;                                \
      p = (T)1.115787767802518096e-17 + p * w;                                 \
      p = (T)-1.333171662854620906e-16 + p * w;                                \
      p = (T)2.0972767875968561637e-17 + p * w;                                \
      p = (T)6.6376381343583238325e-15 + p * w;                                \
      p = (T)-4.0545662729752068639e-14 + p * w;                               \
      p = (T)-8.1519341976054721522e-14 + p * w;                               \
      p = (T)2.6335093153082322977e-12 + p * w;                                \
      p = (T)-1.2975133253453532498e-11 + p * w;                               \
      p = (T)-5.4154120542946279317e-11 + p * w;                               \
      p = (T)1.051212273321532285e-09 + p * w;                                 \
      p = (T)-4.1126339803469836976e-09 + p * w;                               \
      p = (T)-2.9070369957882005086e-08 + p * w;                               \
      p = (T)4.2347877827932403518e-07 + p * w;                                \
      p = (T)-1.3654692000834678645e-06 + p * w;                               \
      p = (T)-1.3882523362786468719e-05 + p * w;                               \
      p = (T)0.0001867342080340571352 + p * w;                                 \
      p = (T)-0.00074070253416626697512 + p * w;                               \
      p = (T)-0.0060336708714301490533 + p * w;                                \
      p = (T)0.24015818242558961693 + p * w;                                   \
      p = (T)1.6536545626831027356 + p * w;                                    \
    } else if (w < (T)16.0) {                                                  \
      w = sqrt(w) - (T)3.25;                                                   \
      p = (T)2.2137376921775787049e-09;                                        \
      p = (T)9.0756561938885390979e-08 + p * w;                                \
      p = (T)-2.7517406297064545428e-07 + p * w;                               \
      p = (T)1.8239629214389227755e-08 + p * w;                                \
      p = (T)1.5027403968909827627e-06 + p * w;                                \
      p = (T)-4.013867526981545969e-06 + p * w;                                \
      p = (T)2.9234449089955446044e-06 + p * w;                                \
      p = (T)1.2475304481671778723e-05 + p * w;                                \
      p = (T)-4.7318229009055733981e-05 + p * w;                               \
      p = (T)6.8284851459573175448e-05 + p * w;                                \
      p = (T)2.4031110387097893999e-05 + p * w;                                \
      p = (T)-0.0003550375203628474796 + p * w;                                \
      p = (T)0.00095328937973738049703 + p * w;                                \
      p = (T)-0.0016882755560235047313 + p * w;                                \
      p = (T)0.0024914420961078508066 + p * w;                                 \
      p = (T)-0.0037512085075692412107 + p * w;                                \
      p = (T)0.005370914553590063617 + p * w;                                  \
      p = (T)1.0052589676941592334 + p * w;                                    \
      p = (T)3.0838856104922207635 + p * w;                                    \
    } else {                                                                   \
      w = sqrt(w) - (T)5.0;                                                    \
      p = (T)-2.7109920616438573243e-11;                                       \
      p = (T)-2.5556418169965252055e-10 + p * w;                               \
      p = (T)1.5076572693500548083e-09 + p * w;                                \
      p = (T)-3.7894654401267369937e-09 + p * w;                               \
      p = (T)7.6157012080783393804e-09 + p * w;                                \
      p = (T)-1.4960026627149240478e-08 + p * w;                               \
      p = (T)2.9147953450901080826e-08 + p * w;                                \
      p = (T)-6.7711997758452339498e-08 + p * w;                               \
      p = (T)2.2900482228026654717e-07 + p * w;                                \
      p = (T)-9.9298272942317002539e-07 + p * w;                               \
      p = (T)4.5260625972231537039e-06 + p * w;                                \
      p = (T)-1.9681778105531670567e-05 + p * w;                               \
      p = (T)7.5995277030017761139e-05 + p * w;                                \
      p = (T)-0.00021503011930044477347 + p * w;                               \
      p = (T)-0.00013871931833623122026 + p * w;                               \
      p = (T)1.0103004648645343977 + p * w;                                    \
      p = (T)4.8499064014085844221 + p * w;                                    \
    }                                                                          \
    return p;                                                                  \
  }                                                                            \
  /* erfcinv(y) for y in (0, 2); Newton on erfc from Giles' start. */          \
  static T vk_erfcinv_##S(T y) {                                               \
    if (!(y > (T)0 && y < (T)2))                                               \
      return y == (T)0 ? (T)INFINITY : y == (T)2 ? -(T)INFINITY : (T)NAN;      \
    T x = giles_##S(-log(y * ((T)2 - y))) * ((T)1 - y);                        \
    for (int i = 0; i < 2; ++i) {                                              \
      T d = (T)1.1283791670955126 * exp(-x * x);                               \
      if (d == (T)0) /* The tail start is as good as it gets. */               \
        break;                                                                 \
      x += (erfc(x) - y) / d;                                                  \
    }                                                                          \
    return x;                                                                  \
  }                                                                            \
  T __ocml_erfcinv_##S(T y) { return vk_erfcinv_##S(y); }                      \
  T __ocml_erfinv_##S(T x) {                                                   \
    if (fabs(x) < (T)0.5) {                                                    \
      T r = giles_##S(-log(((T)1 - x) * ((T)1 + x))) * x;                      \
      for (int i = 0; i < 2; ++i)                                              \
        r -= (erf(r) - x) / ((T)1.1283791670955126 * exp(-r * r));             \
      return r;                                                                \
    }                                                                          \
    T r = vk_erfcinv_##S((T)1 - fabs(x));                                      \
    return x < (T)0 ? -r : r;                                                  \
  }                                                                            \
  T __ocml_ncdf_##S(T x) { return (T)0.5 * erfc(-x * (T)M_SQRT1_2); }          \
  T __ocml_ncdfinv_##S(T p) { return -(T)M_SQRT2 * vk_erfcinv_##S((T)2 * p); } \
  T __ocml_rcbrt_##S(T x) { return (T)1 / cbrt(x); }                           \
  /* erfcx(x) for x >= 0. */                                                   \
  static T vk_erfcx_##S(T x) {                                                 \
    if (x > (T)(sizeof(T) == 4 ? 9 : 10)) {                                    \
      /* Asymptotic series; exp(x*x) would overflow first. */                  \
      T r = (T)1 / (x * x);                                                    \
      return (T)M_2_SQRTPI * (T)0.5 / x *                                      \
             ((T)1 + r * ((T)-0.5 + r * ((T)0.75 + r * (T)-1.875)));           \
    }                                                                          \
    T hi = x * x, lo = fma(x, x, -hi);                                         \
    return exp(hi) * ((T)1 + lo) * erfc(x);                                    \
  }                                                                            \
  T __ocml_erfcx_##S(T x) {                                                    \
    return x < (T)0 ? (T)2 * exp(x * x) - vk_erfcx_##S(-x) : vk_erfcx_##S(x);  \
  }

GEN_SPECIALS(float, f32)
GEN_SPECIALS(double, f64)

// OCML half-precision and classification entry points, on OpenCL builtins.
#define HALF_UNARY(N)                                                          \
  half __ocml_##N##_f16(half x) { return N(x); }                               \
  half2 __ocml_##N##_2f16(half2 x) { return N(x); }
HALF_UNARY(ceil)
HALF_UNARY(cos)
HALF_UNARY(exp)
HALF_UNARY(exp10)
HALF_UNARY(exp2)
HALF_UNARY(fabs)
HALF_UNARY(floor)
HALF_UNARY(log)
HALF_UNARY(log10)
HALF_UNARY(log2)
HALF_UNARY(rint)
HALF_UNARY(sin)
HALF_UNARY(sqrt)
HALF_UNARY(trunc)
// libclc has no half rsqrt for Vulkan.
half __ocml_rsqrt_f16(half x) { return (half)1 / sqrt(x); }
half2 __ocml_rsqrt_2f16(half2 x) { return (half)1 / sqrt(x); }
half __ocml_fma_f16(half a, half b, half c) { return fma(a, b, c); }
half2 __ocml_fma_2f16(half2 a, half2 b, half2 c) { return fma(a, b, c); }
half __ocml_fmax_f16(half a, half b) { return fmax(a, b); }
half __ocml_fmin_f16(half a, half b) { return fmin(a, b); }
// The OpenCL isnan(float) has the mangled name of HIP's isnan(float), which
// calls these; the clang builtins avoid the recursion.
int __ocml_isnan_f16(half x) { return __builtin_isnan(x); }
int __ocml_isinf_f16(half x) { return __builtin_isinf(x); }
// OpenCL vector relationals return -1 for true; HIP's callers expect 1.
short2 __ocml_isnan_2f16(half2 x) { return -isnan(x); }
short2 __ocml_isinf_2f16(half2 x) { return -isinf(x); }
int __ocml_isnan_f32(float x) { return __builtin_isnan(x); }
int __ocml_isinf_f32(float x) { return __builtin_isinf(x); }
int __ocml_isfinite_f32(float x) { return __builtin_isfinite(x); }

// An infinite coordinate wins over a NaN; the reciprocals scale by 1/4 so a
// length past the type's maximum still has a representable reciprocal.
#define GEN_GEOMETRY(T, S)                                                     \
  T __ocml_len3_##S(T a, T b, T c) {                                           \
    if (isinf(a) || isinf(b) || isinf(c))                                      \
      return (T)INFINITY;                                                      \
    if (isnan(a) || isnan(b) || isnan(c))                                      \
      return (T)NAN;                                                           \
    return hypot(hypot(a, b), c);                                              \
  }                                                                            \
  T __ocml_len4_##S(T a, T b, T c, T d) {                                      \
    if (isinf(a) || isinf(b) || isinf(c) || isinf(d))                          \
      return (T)INFINITY;                                                      \
    if (isnan(a) || isnan(b) || isnan(c) || isnan(d))                          \
      return (T)NAN;                                                           \
    return hypot(hypot(a, b), hypot(c, d));                                    \
  }                                                                            \
  T __ocml_rlen3_##S(T a, T b, T c) {                                          \
    T s = fmax(fabs(a), fmax(fabs(b), fabs(c))) > (T)1 ? (T)0.25 : (T)1;       \
    return s / __ocml_len3_##S(a * s, b * s, c * s);                           \
  }                                                                            \
  T __ocml_rlen4_##S(T a, T b, T c, T d) {                                     \
    T s = fmax(fmax(fabs(a), fabs(b)), fmax(fabs(c), fabs(d))) > (T)1          \
              ? (T)0.25                                                        \
              : (T)1;                                                          \
    return s / __ocml_len4_##S(a * s, b * s, c * s, d * s);                    \
  }                                                                            \
  T __ocml_rhypot_##S(T x, T y) {                                              \
    T s = fmax(fabs(x), fabs(y)) > (T)1 ? (T)0.25 : (T)1;                      \
    return s / hypot(x * s, y * s);                                            \
  }                                                                            \
  T __ocml_scalbn_##S(T x, int n) { return ldexp(x, n); }
GEN_GEOMETRY(float, f32)
GEN_GEOMETRY(double, f64)

// scalb(x, y) for an integral y; NaN otherwise, as in C.
#define GEN_SCALB(T, S)                                                        \
  T __ocml_scalb_##S(T x, T y) {                                               \
    if (__builtin_isnan(x) || __builtin_isnan(y))                              \
      return x + y;                                                            \
    if (__builtin_isinf(y))                                                    \
      return y > 0 ? x * y : x / -y;                                           \
    if (y != trunc(y))                                                         \
      return NAN;                                                              \
    return ldexp(x, (int)clamp(y, (T)-100000, (T)100000));                     \
  }
GEN_SCALB(float, f32)
GEN_SCALB(double, f64)

// Bessel functions: the rational approximations of Numerical Recipes 6.5-6.6
// (after Abramowitz and Stegun). Single-precision quality only, poor near the
// zeros and for large arguments (chipStar#1751).
#define GEN_BESSEL(T, S)                                                       \
  T __ocml_j0_##S(T x) {                                                       \
    if (isinf(x) || x == (T)0)                                                 \
      return x == (T)0 ? (T)1 : (T)0;                                          \
    T ax = fabs(x);                                                            \
    if (ax < (T)8) {                                                           \
      T y = x * x;                                                             \
      T a = (T)57568490574.0 +                                                 \
            y * ((T)-13362590354.0 +                                           \
                 y * ((T)651619640.7 +                                         \
                      y * ((T)-11214424.18 +                                   \
                           y * ((T)77392.33017 + y * (T)-184.9052456))));      \
      T b = (T)57568490411.0 +                                                 \
            y * ((T)1029532985.0 +                                             \
                 y * ((T)9494680.718 +                                         \
                      y * ((T)59272.64853 + y * ((T)267.8532712 + y))));       \
      return a / b;                                                            \
    }                                                                          \
    T z = (T)8 / ax, y = z * z, xx = ax - (T)0.785398164;                      \
    T a = (T)1 + y * ((T)-0.1098628627e-2 +                                    \
                      y * ((T)0.2734510407e-4 +                                \
                           y * ((T)-0.2073370639e-5 + y * (T)0.2093887211e-6))); \
    T b = (T)-0.1562499995e-1 +                                                \
          y * ((T)0.1430488765e-3 +                                            \
               y * ((T)-0.6911147651e-5 +                                      \
                    y * ((T)0.7621095161e-6 - y * (T)0.934935152e-7)));        \
    return sqrt((T)0.636619772 / ax) * (cos(xx) * a - z * sin(xx) * b);        \
  }                                                                            \
  T __ocml_j1_##S(T x) {                                                       \
    if (isinf(x))                                                              \
      return copysign((T)0, x);                                                \
    T ax = fabs(x);                                                            \
    if (ax < (T)8) {                                                           \
      T y = x * x;                                                             \
      T a = x * ((T)72362614232.0 +                                            \
                 y * ((T)-7895059235.0 +                                       \
                      y * ((T)242396853.1 +                                    \
                           y * ((T)-2972611.439 +                              \
                                y * ((T)15704.48260 + y * (T)-30.16036606))))); \
      T b = (T)144725228442.0 +                                                \
            y * ((T)2300535178.0 +                                             \
                 y * ((T)18583304.74 +                                         \
                      y * ((T)99447.43394 + y * ((T)376.9991397 + y))));       \
      return a / b;                                                            \
    }                                                                          \
    T z = (T)8 / ax, y = z * z, xx = ax - (T)2.356194491;                      \
    T a = (T)1 + y * ((T)0.183105e-2 +                                         \
                      y * ((T)-0.3516396496e-4 +                               \
                           y * ((T)0.2457520174e-5 + y * (T)-0.240337019e-6))); \
    T b = (T)0.04687499995 +                                                   \
          y * ((T)-0.2002690873e-3 +                                           \
               y * ((T)0.8449199096e-5 +                                       \
                    y * ((T)-0.88228987e-6 + y * (T)0.105787412e-6)));         \
    T r = sqrt((T)0.636619772 / ax) * (cos(xx) * a - z * sin(xx) * b);         \
    return x < (T)0 ? -r : r;                                                  \
  }                                                                            \
  T __ocml_y0_##S(T x) {                                                       \
    if (x < (T)8) {                                                            \
      if (!(x > (T)0))                                                         \
        return x == (T)0 ? -(T)INFINITY : (T)NAN;                              \
      T y = x * x;                                                             \
      T a = (T)-2957821389.0 +                                                 \
            y * ((T)7062834065.0 +                                             \
                 y * ((T)-512359803.6 +                                        \
                      y * ((T)10879881.29 +                                    \
                           y * ((T)-86327.92757 + y * (T)228.4622733))));      \
      T b = (T)40076544269.0 +                                                 \
            y * ((T)745249964.8 +                                              \
                 y * ((T)7189466.438 +                                         \
                      y * ((T)47447.26470 + y * ((T)226.1030244 + y))));       \
      return a / b + (T)0.636619772 * __ocml_j0_##S(x) * log(x);               \
    }                                                                          \
    if (isinf(x))                                                              \
      return (T)0;                                                             \
    T z = (T)8 / x, y = z * z, xx = x - (T)0.785398164;                        \
    T a = (T)1 + y * ((T)-0.1098628627e-2 +                                    \
                      y * ((T)0.2734510407e-4 +                                \
                           y * ((T)-0.2073370639e-5 + y * (T)0.2093887211e-6))); \
    T b = (T)-0.1562499995e-1 +                                                \
          y * ((T)0.1430488765e-3 +                                            \
               y * ((T)-0.6911147651e-5 +                                      \
                    y * ((T)0.7621095161e-6 - y * (T)0.934935152e-7)));        \
    return sqrt((T)0.636619772 / x) * (sin(xx) * a + z * cos(xx) * b);        \
  }                                                                            \
  T __ocml_y1_##S(T x) {                                                       \
    if (x < (T)8) {                                                            \
      if (!(x > (T)0))                                                         \
        return x == (T)0 ? -(T)INFINITY : (T)NAN;                              \
      T y = x * x;                                                             \
      T a = x * ((T)-0.4900604943e13 +                                         \
                 y * ((T)0.1275274390e13 +                                     \
                      y * ((T)-0.5153438139e11 +                               \
                           y * ((T)0.7349264551e9 +                            \
                                y * ((T)-0.4237922726e7 +                      \
                                     y * (T)0.8511937935e4)))));               \
      T b = (T)0.2499580570e14 +                                               \
            y * ((T)0.4244419664e12 +                                          \
                 y * ((T)0.3733650367e10 +                                     \
                      y * ((T)0.2245904002e8 +                                 \
                           y * ((T)0.1020426050e6 +                            \
                                y * ((T)0.3549632885e3 + y)))));               \
      return a / b +                                                           \
             (T)0.636619772 * (__ocml_j1_##S(x) * log(x) - (T)1 / x);          \
    }                                                                          \
    if (isinf(x))                                                              \
      return (T)0;                                                             \
    T z = (T)8 / x, y = z * z, xx = x - (T)2.356194491;                        \
    T a = (T)1 + y * ((T)0.183105e-2 +                                         \
                      y * ((T)-0.3516396496e-4 +                               \
                           y * ((T)0.2457520174e-5 + y * (T)-0.240337019e-6))); \
    T b = (T)0.04687499995 +                                                   \
          y * ((T)-0.2002690873e-3 +                                           \
               y * ((T)0.8449199096e-5 +                                       \
                    y * ((T)-0.88228987e-6 + y * (T)0.105787412e-6)));         \
    return sqrt((T)0.636619772 / x) * (sin(xx) * a + z * cos(xx) * b);        \
  }                                                                            \

GEN_BESSEL(float, f32)
GEN_BESSEL(double, f64)

#define GEN_BESSEL_I(T, S)                                                     \
  T __ocml_i0_##S(T x) {                                                       \
    T ax = fabs(x);                                                            \
    if (ax < (T)3.75) {                                                        \
      T y = (x / (T)3.75) * (x / (T)3.75);                                     \
      return (T)1 +                                                            \
             y * ((T)3.5156229 +                                               \
                  y * ((T)3.0899424 +                                          \
                       y * ((T)1.2067492 +                                     \
                            y * ((T)0.2659732 +                                \
                                 y * ((T)0.360768e-1 + y * (T)0.45813e-2))))); \
    }                                                                          \
    if (isinf(x))                                                              \
      return (T)INFINITY;                                                      \
    /* exp(ax) in halves, so it overflows only with the result. */             \
    T y = (T)3.75 / ax, e = exp(ax * (T)0.5);                                  \
    return e * (e / sqrt(ax) *                                                 \
                ((T)0.39894228 +                                               \
            y * ((T)0.1328592e-1 +                                             \
                 y * ((T)0.225319e-2 +                                         \
                      y * ((T)-0.157565e-2 +                                   \
                           y * ((T)0.916281e-2 +                               \
                                y * ((T)-0.2057706e-1 +                        \
                                     y * ((T)0.2635537e-1 +                    \
                                          y * ((T)-0.1647633e-1 +              \
                                               y * (T)0.392377e-2)))))))));    \
  }                                                                            \
  T __ocml_i1_##S(T x) {                                                       \
    T ax = fabs(x), r;                                                         \
    if (ax < (T)3.75) {                                                        \
      T y = (x / (T)3.75) * (x / (T)3.75);                                     \
      r = ax * ((T)0.5 +                                                       \
                y * ((T)0.87890594 +                                           \
                     y * ((T)0.51498869 +                                      \
                          y * ((T)0.15084934 +                                 \
                               y * ((T)0.2658733e-1 +                          \
                                    y * ((T)0.301532e-2 +                      \
                                         y * (T)0.32411e-3))))));              \
    } else if (isinf(x)) {                                                     \
      r = (T)INFINITY;                                                         \
    } else {                                                                   \
      T y = (T)3.75 / ax, e = exp(ax * (T)0.5);                                \
      r = (T)0.2282967e-1 +                                                    \
          y * ((T)-0.2895312e-1 + y * ((T)0.1787654e-1 - y * (T)0.420059e-2)); \
      r = (T)0.39894228 +                                                      \
          y * ((T)-0.3988024e-1 +                                              \
               y * ((T)-0.362018e-2 +                                          \
                    y * ((T)0.163801e-2 + y * ((T)-0.1031555e-1 + y * r))));   \
      r = e * (e / sqrt(ax) * r);                                              \
    }                                                                          \
    return copysign(r, x);                                                     \
  }
GEN_BESSEL_I(float, f32)

// I0 and I1 in double: the power series up to 30, then the asymptotic
// expansion (Abramowitz and Stegun 9.6.10 and 9.7.1).
static double vk_bessel_i(double x, int nu) {
  double ax = fabs(x), t, s;
  if (ax > 1000) // I0 and I1 overflow near 713; avoids inf / inf below.
    return INFINITY;
  if (ax <= 30) {
    double q = 0.25 * ax * ax;
    t = s = nu ? 0.5 * ax : 1.0;
    for (int k = 1; t > s * 0x1p-60; ++k) {
      t *= q / (k * (k + nu));
      s += t;
    }
    return s;
  }
  t = s = 1.0;
  for (int k = 1; k < 40 && fabs(t) > s * 0x1p-60; ++k) {
    t *= ((2 * k - 1) * (2 * k - 1) - 4 * nu) / (8.0 * k * ax);
    s += t;
  }
  // exp(ax) in halves, so it overflows only with the result.
  double e = exp(0.5 * ax);
  return e * (e / sqrt(2 * M_PI * ax) * s);
}
double __ocml_i0_f64(double x) { return vk_bessel_i(x, 0); }
double __ocml_i1_f64(double x) { return copysign(vk_bessel_i(x, 1), x); }
