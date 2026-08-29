/*
 * Copyright (c) 2021-23 chipStar developers
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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#if __cplusplus >= 201103L
#include <thread>
#endif

#include <hip/driver_types.h>
#include <hip/spirv_hip_host_defines.h>
#include "chipStarConfig.hh"

#if defined(__clang__) && defined(__HIP__)
#include "spirv_hip_devicelib.hh"

#define uint uint32_t

#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define HIP_SYMBOL(X) X

#define HIP_DYNAMIC_SHARED(type, var) extern __shared__ type var[];

#define HIP_DYNAMIC_SHARED_ATTRIBUTE

extern "C" {
// A global flag included in all HIP device modules for signaling
// abort request.
__attribute__((weak)) __device__ int32_t __chipspv_abort_called;
 
// The message of a failed device-side assertion, in the format
// _cl_assert_fail_print prints it. Device printf only reaches stdout, so
// __assert_fail also records the message here for the runtime to copy to the
// host and write to stderr when it services the abort request
// (handleAbortRequest); stderr is where ROCm reports it and where gtest death
// tests look for it. Claimed is taken atomically by the first work-item that
// records a message so concurrent failures do not interleave in Text. The
// runtime treats the variable as raw bytes and reads Text at offset
// sizeof(int) (ChipDeviceAbortMsgTextOffset in src/common.hh).
struct __chipspv_abort_msg_t {
  int Claimed;
  char Text[508];
};
__attribute__((weak)) __device__ __chipspv_abort_msg_t __chipspv_abort_msg;

// Global pointer for the device heap for device-side malloc/free.
// Gated behind CHIP_ENABLE_DEVICE_PROGRAM_SCOPE_GLOBALS: this is a program-scope
// global which some OpenCL drivers (e.g. rusticl/radeonsi) cannot consume
// (issue #1279).
#ifdef CHIP_ENABLE_DEVICE_PROGRAM_SCOPE_GLOBALS
__attribute__((weak)) __device__ void* __chipspv_device_heap;
#endif

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

extern "C" __device__ int printf(const char *fmt, ...)
    __attribute__((format(printf, 1, 2)));
extern "C" __device__ void abort();

static inline __device__ void __chipspv_abort_msg_append(unsigned &Pos,
                                                         const char *S) {
  while (*S && Pos < sizeof(__chipspv_abort_msg.Text) - 1)
    __chipspv_abort_msg.Text[Pos++] = *S++;
}

// Records a failed assertion in __chipspv_abort_msg for the runtime to report
// on stderr. Only the first work-item to claim the buffer writes it; the text
// is bounded by the buffer and NUL-terminated.
static inline __device__ void
__chipspv_record_assert_msg(const char *file, unsigned int line,
                            const char *function, const char *assertion) {
  if (atomicCAS(&__chipspv_abort_msg.Claimed, 0, 1) != 0)
    return;

  // Decimal digits of the line number, least significant first.
  char Digits[10];
  unsigned NumDigits = 0;
  do {
    Digits[NumDigits++] = '0' + line % 10;
    line /= 10;
  } while (line);
  char LineStr[11];
  unsigned I = 0;
  while (NumDigits)
    LineStr[I++] = Digits[--NumDigits];
  LineStr[I] = 0;

  unsigned Pos = 0;
  __chipspv_abort_msg_append(Pos, file);
  __chipspv_abort_msg_append(Pos, ":");
  __chipspv_abort_msg_append(Pos, LineStr);
  __chipspv_abort_msg_append(Pos, ": ");
  __chipspv_abort_msg_append(Pos, function);
  __chipspv_abort_msg_append(Pos, ": Device-side assertion `");
  __chipspv_abort_msg_append(Pos, assertion);
  __chipspv_abort_msg_append(Pos, "' failed.");
  __chipspv_abort_msg.Text[Pos] = 0;
}

// The assert part mimiced from amd_device_functions.h of amdhip.  We
// assume assert.h defines assert such that it calls __assert_fail
// when it fails. Some users forward declares the __assert_fail, with
// a signature copied from the AMD's headers, in their
// applications. If we don't reflect the function signature the
// compiler may give obscure warnings about it.

extern "C" {
// Message printing is delegated to _cl_assert_fail_print in chipStar's device
// library (bitcode/_cl_print_str.cl). Printing there (OpenCL C) keeps the
// printf format strings in the constant address space; emitting them from a
// HIP C++ device function would place the literals in the generic address
// space and force SPV_EXT_relaxed_printf_string_address_space, which the Intel
// runtime rejects at clBuildProgram. Aborting stays here so abort() resolves
// to the flag-based __chipspv_abort path (handled by the HipAbort pass).
__device__ void _cl_assert_fail_print(const char *file, unsigned int line,
                                      const char *function,
                                      const char *assertion);
#if defined(_WIN32) || defined(_WIN64)
__device__ __attribute__((noinline)) __attribute__((weak)) void
_wassert(const wchar_t *_msg, const wchar_t *_file, unsigned _line)
    __attribute__((
        warning("device side assert is not supported on Windows yet"))) {
  // FIXME: Need `wchar_t` support to generate assertion message.
  abort();
}
#elif defined(__APPLE__)
// On macOS, assert() expands to __assert_rtn instead of __assert_fail.
__device__ __attribute__((noinline)) __attribute__((weak)) void
__assert_rtn(const char *function, const char *file, int line,
             const char *assertion) {
  _cl_assert_fail_print(file, line, function, assertion);
  __chipspv_record_assert_msg(file, line, function, assertion);
  abort();
}
#else  // defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
__device__ __attribute__((noinline)) __attribute__((weak)) void
__assert_fail(const char *assertion, const char *file, unsigned int line,
              const char *function) {
  _cl_assert_fail_print(file, line, function, assertion);
  __chipspv_record_assert_msg(file, line, function, assertion);
  abort();
}
#endif // defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)

// Clang materialises a reference to __cxa_pure_virtual in the vtable of every
// class that still has an unoverridden pure virtual member. The symbol has to
// resolve on the device even when the slot can never be reached, otherwise the
// SPIR-V consumer rejects the whole module with
//   unresolved external symbol #__cxa_pure_virtual in data segment
// and every kernel in the program fails to build. ROCm supplies it from its
// device libs; on the SPIR-V platform there is no equivalent, so define it here
// with the same semantics: calling a pure virtual function is a hard error.
// No printf here on purpose: it would put a device printf into every
// translation unit, and each one costs HipVerify's pre-lowering checkpoints a
// crashing llvm-spirv run in debug builds (about 13 s per TU).
__device__ __attribute__((noinline)) __attribute__((weak)) void
__cxa_pure_virtual() {
  abort();
}
} // extern "C"
#endif // defined(__clang__) && defined(__HIP__)

/*************************************************************************************************/

typedef enum {
  EVENT_STATUS_INIT = 5,
  EVENT_STATUS_RECORDING,
  EVENT_STATUS_RECORDED
} event_status_e;

#endif
