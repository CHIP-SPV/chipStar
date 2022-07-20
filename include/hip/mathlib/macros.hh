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
#endif

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

#endif // include guard