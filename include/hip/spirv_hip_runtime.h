/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef SPIRV_HIP_RUNTIME_H
#define SPIRV_HIP_RUNTIME_H

#ifndef __HIP_PLATFORM_SPIRV__
#define __HIP_PLATFORM_SPIRV__
#endif

#include "chipStarConfig.hh"

#ifdef __cplusplus
#include <cmath>
#include <cstdint>
#endif

#if defined(__clang__) && defined(__HIP__)
#define __HIP_CLANG_ONLY__ 1
#else
#define __HIP_CLANG_ONLY__ 0
#endif

#include <hip/hip_runtime_api.h>
#include <hip/spirv_hip.hh>
#include <hip/spirv_hip_vector_types.h>
#include <hip/spirv_math_fwd.h>
#include <hip/spirv_hip_host_math.h>
#include <hip/spirv_texture_functions.h>
#include <hip/spirv_hip_ldg.h>

// Feature tests:
#if (defined(__HCC_ACCELERATOR__) && (__HCC_ACCELERATOR__ != 0)) ||            \
    __HIP_DEVICE_COMPILE__
// Device compile and not host compile:

// 32-bit Atomics:
#define __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__ (1)
#define __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__ (0)
#define __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__ (1)
#define __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__ (0)
#define __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__ (0)

// 64-bit Atomics:
#define __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__ (1)
#define __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__ (1)

// Doubles
#define __HIP_ARCH_HAS_DOUBLES__ (1)

// warp cross-lane operations:
#define __HIP_ARCH_HAS_WARP_VOTE__ (0)
#define __HIP_ARCH_HAS_WARP_BALLOT__ (0)
#define __HIP_ARCH_HAS_WARP_SHUFFLE__ (0)
#define __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__ (0)

// sync
#define __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__ (0)
#define __HIP_ARCH_HAS_SYNC_THREAD_EXT__ (0)

// misc
#define __HIP_ARCH_HAS_SURFACE_FUNCS__ (0)
#define __HIP_ARCH_HAS_3DGRID__ (0)
#define __HIP_ARCH_HAS_DYNAMIC_PARALLEL__ (0)

#endif /* Device feature flags */

#endif
