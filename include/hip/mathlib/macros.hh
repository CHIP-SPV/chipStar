#ifndef HIP_INCLUDE_MATHLIB_MACROS_H
#define HIP_INCLUDE_MATHLIB_MACROS_H

#include <hip/spirv_math_fwd.h>
#include <algorithm>
#include <limits>

#define NOOPT __attribute__((optnone))

#if defined(__HIP_DEVICE_COMPILE__)
#define __DEVICE__ static __device__
#define EXPORT static inline __device__
#define OVLD __attribute__((overloadable)) __device__
#define NON_OVLD __device__
#define GEN_NAME(N) opencl_##N
#define GEN_NAME2(N, S) opencl_##N##_##S
#else
#define __DEVICE__ extern __device__
#define EXPORT extern __device__
#define NON_OVLD
#define OVLD
#define GEN_NAME(N) N
#define GEN_NAME2(N, S) N
#endif // __HIP_DEVICE_COMPILE__

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

#if defined(__HIP_DEVICE_COMPILE__)
typedef _Float16 api_half;
typedef _Float16 api_half2 __attribute__((ext_vector_type(2)));
#else
typedef short api_half;
typedef short api_half2 __attribute__((ext_vector_type(2)));
#endif // __HIP_DEVICE_COMPILE__

#if defined(__HIP_DEVICE_COMPILE__)

#define DEFOPENCL1F(NAME)                                                      \
  extern "C" {                                                                 \
  float NON_OVLD GEN_NAME2(NAME, f)(float f);                                  \
  double NON_OVLD GEN_NAME2(NAME, d)(double f);                                \
  api_half NON_OVLD GEN_NAME2(NAME, h)(api_half f);                            \
  api_half2 NON_OVLD GEN_NAME2(NAME, h2)(api_half2 f);                         \
  }                                                                            \
  EXPORT float NAME##f(float x) { return GEN_NAME2(NAME, f)(x); }              \
  EXPORT double NAME(double x) { return GEN_NAME2(NAME, d)(x); }               \
  EXPORT api_half NAME##_h(api_half x) { return GEN_NAME2(NAME, h)(x); }       \
  EXPORT api_half2 NAME##_2h(api_half2 x) { return GEN_NAME2(NAME, h2)(x); }

#define DEFOPENCL2F(NAME)                                                      \
  extern "C" {                                                                 \
  float NON_OVLD GEN_NAME2(NAME, f)(float x, float y);                         \
  double NON_OVLD GEN_NAME2(NAME, d)(double x, double y);                      \
  api_half NON_OVLD GEN_NAME2(NAME, h)(api_half x, api_half y);                \
  api_half2 NON_OVLD GEN_NAME2(NAME, h2)(api_half2 x, api_half2 y);            \
  }                                                                            \
  EXPORT float NAME##f(float x, float y) { return GEN_NAME2(NAME, f)(x, y); }  \
  EXPORT double NAME(double x, double y) { return GEN_NAME2(NAME, d)(x, y); }  \
  EXPORT api_half NAME##_h(api_half x, api_half y) {                           \
    return GEN_NAME2(NAME, h)(x, y);                                           \
  }                                                                            \
  EXPORT api_half2 NAME##_2h(api_half2 x, api_half2 y) {                       \
    return GEN_NAME2(NAME, h2)(x, y);                                          \
  }

#define DEFOPENCL3F(NAME)                                                      \
  extern "C" {                                                                 \
  float NON_OVLD GEN_NAME2(NAME, f)(float x, float y, float z);                \
  double NON_OVLD GEN_NAME2(NAME, d)(double x, double y, double z);            \
  api_half NON_OVLD GEN_NAME2(NAME, h)(api_half x, api_half y, api_half z);    \
  api_half2 NON_OVLD GEN_NAME2(NAME, h2)(api_half2 x, api_half2 y,             \
                                         api_half2 z);                         \
  }                                                                            \
  EXPORT float NAME##f(float x, float y, float z) {                            \
    return GEN_NAME2(NAME, f)(x, y, z);                                        \
  }                                                                            \
  EXPORT double NAME(double x, double y, double z) {                           \
    return GEN_NAME2(NAME, d)(x, y, z);                                        \
  }                                                                            \
  EXPORT api_half NAME##_h(api_half x, api_half y, api_half z) {               \
    return GEN_NAME2(NAME, h)(x, y, z);                                        \
  }                                                                            \
  EXPORT api_half2 NAME##_2h(api_half2 x, api_half2 y, api_half2 z) {          \
    return GEN_NAME2(NAME, h2)(x, y, z);                                       \
  }

#define DEFOPENCL4F(NAME)                                                      \
  extern "C" {                                                                 \
  float NON_OVLD GEN_NAME2(NAME, f)(float x, float y, float z, float w);       \
  double NON_OVLD GEN_NAME2(NAME, d)(double x, double y, double z, double w);  \
  api_half NON_OVLD GEN_NAME2(NAME, h)(api_half x, api_half y, api_half z,     \
                                       api_half w);                            \
  api_half2 NON_OVLD GEN_NAME2(NAME, h2)(api_half2 x, api_half2 y,             \
                                         api_half2 z, api_half2 w);            \
  }                                                                            \
  EXPORT float NAME##f(float x, float y, float z, float w) {                   \
    return GEN_NAME2(NAME, f)(x, y, z, w);                                     \
  }                                                                            \
  EXPORT double NAME(double x, double y, double z, double w) {                 \
    return GEN_NAME2(NAME, d)(x, y, z, w);                                     \
  }                                                                            \
  EXPORT api_half NAME##_h(api_half x, api_half y, api_half z, api_half w) {   \
    return GEN_NAME2(NAME, h)(x, y, z, w);                                     \
  }                                                                            \
  EXPORT api_half2 NAME##_2h(api_half2 x, api_half2 y, api_half2 z,            \
                             api_half2 w) {                                    \
    return GEN_NAME2(NAME, h2)(x, y, z, w);                                    \
  }

#define DEFOPENCL1B(NAME)                                                      \
  extern "C" {                                                                 \
  int NON_OVLD GEN_NAME2(NAME, f)(float f);                                    \
  long NON_OVLD GEN_NAME2(NAME, d)(double f);                                  \
  api_half NON_OVLD GEN_NAME2(NAME, h)(api_half f);                            \
  api_half2 NON_OVLD GEN_NAME2(NAME, h2)(api_half2 f);                         \
  }                                                                            \
  EXPORT bool NAME(float x) { return (bool)GEN_NAME2(NAME, f)(x); }            \
  EXPORT bool NAME(double x) { return (bool)GEN_NAME2(NAME, d)(x); }           \
  EXPORT api_half NAME##_h(api_half x) { return GEN_NAME2(NAME, h)(x); }       \
  EXPORT api_half2 NAME##_2h(api_half2 x) { return GEN_NAME2(NAME, h2)(x); }

#define DEFOPENCL1INT(NAME)                                                    \
  extern "C" {                                                                 \
  int NON_OVLD GEN_NAME2(NAME, f)(float f);                                    \
  int NON_OVLD GEN_NAME2(NAME, d)(double f);                                   \
  int NON_OVLD GEN_NAME2(NAME, h)(api_half f);                                 \
  }                                                                            \
  EXPORT int NAME##f(float x) { return GEN_NAME2(NAME, f)(x); }                \
  EXPORT int NAME(double x) { return GEN_NAME2(NAME, d)(x); }                  \
  EXPORT int NAME##_h(api_half x) { return GEN_NAME2(NAME, h)(x); }

#define DEFOPENCL1LL(NAME)                                                     \
  extern "C" {                                                                 \
  int64_t NON_OVLD GEN_NAME2(LL##NAME, f)(float f);                            \
  int64_t NON_OVLD GEN_NAME2(LL##NAME, d)(double f);                           \
  }                                                                            \
  EXPORT long int l##NAME##f(float x) {                                        \
    return (long int)GEN_NAME2(LL##NAME, f)(x);                                \
  }                                                                            \
  EXPORT long int l##NAME(double x) {                                          \
    return (long int)GEN_NAME2(LL##NAME, d)(x);                                \
  }                                                                            \
  EXPORT long long int ll##NAME##f(float x) {                                  \
    return (long long int)GEN_NAME2(LL##NAME, f)(x);                           \
  }                                                                            \
  EXPORT long long int ll##NAME(double x) {                                    \
    return (long long int)GEN_NAME2(LL##NAME, d)(x);                           \
  }

#define DEFOPENCL1F_NATIVE(NAME)                                               \
  extern "C" {                                                                 \
  float NON_OVLD GEN_NAME2(NAME##_native, f)(float f);                         \
  }                                                                            \
  EXPORT float __##NAME##f(float x) { return GEN_NAME2(NAME##_native, f)(x); }

#define DEFOPENCL2F_NATIVE(NAME)                                               \
  extern "C" {                                                                 \
  float NON_OVLD GEN_NAME2(NAME##_native, f)(float x, float y);                \
  }                                                                            \
  EXPORT float __##NAME##f(float x, float y) {                                 \
    return GEN_NAME2(NAME##_native, f)(x, y);                                  \
  }

#define FAKE_ROUNDINGS2(NAME, CODE)                                            \
  EXPORT float __f##NAME##_rd(float x, float y) { return CODE; }               \
  EXPORT float __f##NAME##_rn(float x, float y) { return CODE; }               \
  EXPORT float __f##NAME##_ru(float x, float y) { return CODE; }               \
  EXPORT float __f##NAME##_rz(float x, float y) { return CODE; }               \
  EXPORT double __d##NAME##_rd(double x, double y) { return CODE; }            \
  EXPORT double __d##NAME##_rn(double x, double y) { return CODE; }            \
  EXPORT double __d##NAME##_ru(double x, double y) { return CODE; }            \
  EXPORT double __d##NAME##_rz(double x, double y) { return CODE; }

#define FAKE_ROUNDINGS1(NAME, CODE)                                            \
  EXPORT float __f##NAME##_rd(float x) { return CODE; }                        \
  EXPORT float __f##NAME##_rn(float x) { return CODE; }                        \
  EXPORT float __f##NAME##_ru(float x) { return CODE; }                        \
  EXPORT float __f##NAME##_rz(float x) { return CODE; }                        \
  EXPORT double __d##NAME##_rd(double x) { return CODE; }                      \
  EXPORT double __d##NAME##_rn(double x) { return CODE; }                      \
  EXPORT double __d##NAME##_ru(double x) { return CODE; }                      \
  EXPORT double __d##NAME##_rz(double x) { return CODE; }

#define FAKE_ROUNDINGS3(NAME, CODE)                                            \
  EXPORT float __##NAME##f_rd(float x, float y, float z) { return CODE; }      \
  EXPORT float __##NAME##f_rn(float x, float y, float z) { return CODE; }      \
  EXPORT float __##NAME##f_ru(float x, float y, float z) { return CODE; }      \
  EXPORT float __##NAME##f_rz(float x, float y, float z) { return CODE; }      \
  EXPORT double __##NAME##_rd(double x, double y, double z) { return CODE; }   \
  EXPORT double __##NAME##_rn(double x, double y, double z) { return CODE; }   \
  EXPORT double __##NAME##_ru(double x, double y, double z) { return CODE; }   \
  EXPORT double __##NAME##_rz(double x, double y, double z) { return CODE; }

#else

#define DEFOPENCL1F(NAME)                                                      \
  EXPORT float NAME##f(float x);                                               \
  EXPORT double NAME(double x);                                                \
  EXPORT api_half NAME##_h(api_half x);                                        \
  EXPORT api_half2 NAME##_2h(api_half2 x);

#define DEFOPENCL2F(NAME)                                                      \
  EXPORT float NAME##f(float x, float y);                                      \
  EXPORT double NAME(double x, double y);                                      \
  EXPORT api_half NAME##_h(api_half x, api_half y);                            \
  EXPORT api_half2 NAME##_2h(api_half2 x, api_half2 y);

#define DEFOPENCL3F(NAME)                                                      \
  EXPORT float NAME##f(float x, float y, float z);                             \
  EXPORT double NAME(double x, double y, double z);                            \
  EXPORT api_half NAME##_h(api_half x, api_half y, api_half z);                \
  EXPORT api_half2 NAME##_2h(api_half2 x, api_half2 y, api_half2 z);

#define DEFOPENCL4F(NAME)                                                      \
  EXPORT float NAME##f(float x, float y, float z, float w);                    \
  EXPORT double NAME(double x, double y, double z, double w);                  \
  EXPORT api_half NAME##_h(api_half x, api_half y, api_half z, api_half w);    \
  EXPORT api_half2 NAME##_2h(api_half2 x, api_half2 y, api_half2 z,            \
                             api_half2 w);

#define DEFOPENCL1B(NAME)                                                      \
  EXPORT bool NAME(float x);                                                   \
  EXPORT bool NAME(double x);                                                  \
  EXPORT api_half NAME##_h(api_half x);                                        \
  EXPORT api_half2 NAME##_2h(api_half2 x);

#define DEFOPENCL1INT(NAME)                                                    \
  EXPORT int NAME##f(float x);                                                 \
  EXPORT int NAME(double x);                                                   \
  EXPORT int NAME##_h(api_half x);

#define DEFOPENCL1LL(NAME)                                                     \
  EXPORT long int l##NAME##f(float x);                                         \
  EXPORT long int l##NAME(double x);                                           \
  EXPORT long long int ll##NAME##f(float x);                                   \
  EXPORT long long int ll##NAME(double x);

#define DEFOPENCL1F_NATIVE(NAME) EXPORT float __##NAME##f(float x);
#define DEFOPENCL2F_NATIVE(NAME) EXPORT float __##NAME##f(float x, float y);

#define FAKE_ROUNDINGS2(NAME, CODE)                                            \
  EXPORT float __f##NAME##_rd(float x, float y);                               \
  EXPORT float __f##NAME##_rn(float x, float y);                               \
  EXPORT float __f##NAME##_ru(float x, float y);                               \
  EXPORT float __f##NAME##_rz(float x, float y);                               \
  EXPORT double __d##NAME##_rd(double x, double y);                            \
  EXPORT double __d##NAME##_rn(double x, double y);                            \
  EXPORT double __d##NAME##_ru(double x, double y);                            \
  EXPORT double __d##NAME##_rz(double x, double y);

#define FAKE_ROUNDINGS1(NAME, CODE)                                            \
  EXPORT float __f##NAME##_rd(float x);                                        \
  EXPORT float __f##NAME##_rn(float x);                                        \
  EXPORT float __f##NAME##_ru(float x);                                        \
  EXPORT float __f##NAME##_rz(float x);                                        \
  EXPORT double __d##NAME##_rd(double x);                                      \
  EXPORT double __d##NAME##_rn(double x);                                      \
  EXPORT double __d##NAME##_ru(double x);                                      \
  EXPORT double __d##NAME##_rz(double x);

#define FAKE_ROUNDINGS3(NAME, CODE)                                            \
  EXPORT float __##NAME##f_rd(float x, float y, float z);                      \
  EXPORT float __##NAME##f_rn(float x, float y, float z);                      \
  EXPORT float __##NAME##f_ru(float x, float y, float z);                      \
  EXPORT float __##NAME##f_rz(float x, float y, float z);                      \
  EXPORT double __##NAME##_rd(double x, double y, double z);                   \
  EXPORT double __##NAME##_rn(double x, double y, double z);                   \
  EXPORT double __##NAME##_ru(double x, double y, double z);                   \
  EXPORT double __##NAME##_rz(double x, double y, double z);

#endif

#endif // include guard