/*
 * Copyright (c) 2021-22 CHIP-SPV developers
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


#ifndef HIP_H
#define HIP_H

#include <stddef.h>
#include <stdint.h>

#include <hip/driver_types.h>
#include <hip/spirv_hip_host_defines.h>

#if defined(__clang__) && defined(__HIP__)
#include "spirv_hip_devicelib.hh"

#define uint uint32_t

#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define HIP_SYMBOL(X) X

#define HIP_DYNAMIC_SHARED(type, var) __shared__ type var[4294967295];

#define HIP_DYNAMIC_SHARED_ATTRIBUTE

extern "C" {
// A global flag included in all HIP device modules for signaling
// abort request.
__attribute__((weak)) __device__ int32_t __chipspv_abort_called;

__device__ void __chipspv_abort(int32_t *abort_flag);

static inline __device__ void abort() {
  __chipspv_abort(&__chipspv_abort_called);
}
}

typedef int hipLaunchParm;
#define hipLaunchKernelGGLInternal(kernelName, numBlocks, numThreads,          \
                                   memPerBlock, streamId, ...)                 \
  do {                                                                         \
    kernelName<<<(numBlocks), (numThreads), (memPerBlock), (streamId)>>>(      \
        __VA_ARGS__);                                                          \
  } while (0)

#define hipLaunchKernelGGL(kernelName, ...)                                    \
  hipLaunchKernelGGLInternal((kernelName), __VA_ARGS__)

#pragma push_macro("__DEVICE__")
#define __DEVICE__ static __device__ __forceinline__

extern "C" __device__ size_t _Z12get_local_idj(uint);
__DEVICE__ uint __hip_get_thread_idx_x() { return _Z12get_local_idj(0); }
__DEVICE__ uint __hip_get_thread_idx_y() { return _Z12get_local_idj(1); }
__DEVICE__ uint __hip_get_thread_idx_z() { return _Z12get_local_idj(2); }

extern "C" __device__ size_t _Z12get_group_idj(uint);
__DEVICE__ uint __hip_get_block_idx_x() { return _Z12get_group_idj(0); }
__DEVICE__ uint __hip_get_block_idx_y() { return _Z12get_group_idj(1); }
__DEVICE__ uint __hip_get_block_idx_z() { return _Z12get_group_idj(2); }

extern "C" __device__ size_t _Z14get_local_sizej(uint);
__DEVICE__ uint __hip_get_block_dim_x() { return _Z14get_local_sizej(0); }
__DEVICE__ uint __hip_get_block_dim_y() { return _Z14get_local_sizej(1); }
__DEVICE__ uint __hip_get_block_dim_z() { return _Z14get_local_sizej(2); }

extern "C" __device__ size_t _Z14get_num_groupsj(uint);
__DEVICE__ uint __hip_get_grid_dim_x() { return _Z14get_num_groupsj(0); }
__DEVICE__ uint __hip_get_grid_dim_y() { return _Z14get_num_groupsj(1); }
__DEVICE__ uint __hip_get_grid_dim_z() { return _Z14get_num_groupsj(2); }

#define __HIP_DEVICE_BUILTIN(DIMENSION, FUNCTION)                              \
  __declspec(property(get = __get_##DIMENSION)) uint DIMENSION;                \
  __DEVICE__ uint __get_##DIMENSION(void) { return FUNCTION; }

struct __hip_builtin_threadIdx_t {
  __HIP_DEVICE_BUILTIN(x, __hip_get_thread_idx_x());
  __HIP_DEVICE_BUILTIN(y, __hip_get_thread_idx_y());
  __HIP_DEVICE_BUILTIN(z, __hip_get_thread_idx_z());
};

struct __hip_builtin_blockIdx_t {
  __HIP_DEVICE_BUILTIN(x, __hip_get_block_idx_x());
  __HIP_DEVICE_BUILTIN(y, __hip_get_block_idx_y());
  __HIP_DEVICE_BUILTIN(z, __hip_get_block_idx_z());
};

struct __hip_builtin_blockDim_t {
  __HIP_DEVICE_BUILTIN(x, __hip_get_block_dim_x());
  __HIP_DEVICE_BUILTIN(y, __hip_get_block_dim_y());
  __HIP_DEVICE_BUILTIN(z, __hip_get_block_dim_z());
};

struct __hip_builtin_gridDim_t {
  __HIP_DEVICE_BUILTIN(x, __hip_get_grid_dim_x());
  __HIP_DEVICE_BUILTIN(y, __hip_get_grid_dim_y());
  __HIP_DEVICE_BUILTIN(z, __hip_get_grid_dim_z());
};

#undef uint
#undef __HIP_DEVICE_BUILTIN
#pragma pop_macro("__DEVICE__")

extern const __device__ __attribute__((weak))
__hip_builtin_threadIdx_t threadIdx;
extern const __device__ __attribute__((weak)) __hip_builtin_blockIdx_t blockIdx;
extern const __device__ __attribute__((weak)) __hip_builtin_blockDim_t blockDim;
extern const __device__ __attribute__((weak)) __hip_builtin_gridDim_t gridDim;

#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z

#define hipBlockIdx_x blockIdx.x
#define hipBlockIdx_y blockIdx.y
#define hipBlockIdx_z blockIdx.z

#define hipBlockDim_x blockDim.x
#define hipBlockDim_y blockDim.y
#define hipBlockDim_z blockDim.z

#define hipGridDim_x gridDim.x
#define hipGridDim_y gridDim.y
#define hipGridDim_z gridDim.z

#endif // defined(__clang__) && defined(__HIP__)

/* FIXME: Is this the best place for these declarations? */

#ifdef __cplusplus
extern "C" __device__ int printf(const char *fmt, ...)
    __attribute__((format(printf, 1, 2)));
extern "C" __device__ void abort();
#else
extern __device__ int printf(const char *fmt, ...)
    __attribute__((format(printf, 1, 2)));
extern __device__ void abort();
#endif
/*************************************************************************************************/

typedef enum {
  EVENT_STATUS_INIT = 5,
  EVENT_STATUS_RECORDING,
  EVENT_STATUS_RECORDED
} event_status_e;

#endif
