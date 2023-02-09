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

/**
 * @file spirv_spt.h
 * @author Paulius Velesko (pvelesko@pglc.io)
 * @brief Header defining global CHIP classes and functions such as
 * CHIPBackend type pointer Backend which gets initialized at the start of
 * execution.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SPIRV_SPT_H
#define SPIRV_SPT_H

#include "spirv_hip_runtime.h"



/**
 * If HIP_API_PER_THREAD_DEFAULT_STREAM is defined, we map all regular functions to the per-thread stream versions
 * hipMemcpy() -> hipMemcpy_spt()
 * 
 * If HIP_API_PER_THREAD_DEFAULT_STREAM is not defined, we have two versions:
 * hipMemcpy()
 * hipMemcpy_spt()
 * 
 * We must ensure that once CHIP-SPV is compiled, we can still use `hipcc ----default-stream`
 */

#include "spirv_hip_runtime.h"
/// hipStreamPerThread implementation
#if defined(HIP_API_PER_THREAD_DEFAULT_STREAM)
    #define __HIP_STREAM_PER_THREAD
    #define __HIP_API_SPT(api) api ## _spt
#else
    #define __HIP_API_SPT(api) api
#endif

#if defined(__HIP_STREAM_PER_THREAD)
    // Memory APIs
    #define hipMemcpy                     __HIP_API_SPT(hipMemcpy)
    #define hipMemcpyToSymbol             __HIP_API_SPT(hipMemcpyToSymbol)
    #define hipMemcpyFromSymbol           __HIP_API_SPT(hipMemcpyFromSymbol)
    #define hipMemcpy2D                   __HIP_API_SPT(hipMemcpy2D)
    #define hipMemcpy2DToArray            __HIP_API_SPT(hipMemcpy2DToArray)
    #define hipMemcpy2DFromArray          __HIP_API_SPT(hipMemcpy2DFromArray)
    #define hipMemcpy3D                   __HIP_API_SPT(hipMemcpy3D)
    #define hipMemset                     __HIP_API_SPT(hipMemset)
    #define hipMemset2D                   __HIP_API_SPT(hipMemset2D)
    #define hipMemset3D                   __HIP_API_SPT(hipMemset3D)
    #define hipMemcpyAsync                __HIP_API_SPT(hipMemcpyAsync)

    // Stream APIs
    #define hipStreamSynchronize          __HIP_API_SPT(hipStreamSynchronize)
    #define hipStreamQuery                __HIP_API_SPT(hipStreamQuery)
    #define hipStreamGetFlags             __HIP_API_SPT(hipStreamGetFlags)
    #define hipStreamGetPriority          __HIP_API_SPT(hipStreamGetPriority)
    #define hipStreamWaitEvent            __HIP_API_SPT(hipStreamWaitEvent)

    // Event APIs
    #define hipEventRecord               __HIP_API_SPT(hipEventRecord)

    // Launch APIs
    #define hipLaunchKernel               __HIP_API_SPT(hipLaunchKernel)
    #define hipLaunchCooperativeKernel    __HIP_API_SPT(hipLaunchCooperativeKernel)
#endif

hipError_t hipMemcpy_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

hipError_t hipMemcpyToSymbol_spt(const void* symbol, const void* src, size_t sizeBytes,
                             size_t offset, hipMemcpyKind kind);

hipError_t hipMemcpyFromSymbol_spt(void* dst, const void* symbol,size_t sizeBytes,
                               size_t offset, hipMemcpyKind kind);

hipError_t hipMemcpy2D_spt(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                        size_t height, hipMemcpyKind kind);

hipError_t hipMemcpy2DToArray_spt(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind);

hipError_t hipMemcpy2DFromArray_spt( void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                        size_t hOffset, size_t width, size_t height, hipMemcpyKind kind);

hipError_t hipMemcpy3D_spt(const struct hipMemcpy3DParms* p);

hipError_t hipMemset_spt(void* dst, int value, size_t sizeBytes);

hipError_t hipMemset2D_spt(void* dst, size_t pitch, int value, size_t width, size_t height);

hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent );

hipError_t hipMemcpyAsync_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream);

hipError_t hipStreamQuery_spt(hipStream_t stream);

hipError_t hipStreamSynchronize_spt(hipStream_t stream);

hipError_t hipStreamGetPriority_spt(hipStream_t stream, int* priority);

hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event, unsigned int flags);

hipError_t hipEventRecord_spt(hipEvent_t Event, hipStream_t Stream);

hipError_t hipStreamGetFlags_spt(hipStream_t stream, unsigned int* flags);

hipError_t hipLaunchCooperativeKernel_spt(const void* f,
                                      dim3 gridDim, dim3 blockDim,
                                      void **kernelParams, uint32_t sharedMemBytes, hipStream_t hStream);
#ifdef __cplusplus
extern "C" {
#endif
hipError_t hipLaunchKernel_spt(const void* function_address,
                           dim3 numBlocks,
                           dim3 dimBlocks,
                           void** args,
                           size_t sharedMemBytes, hipStream_t stream);
#ifdef __cplusplus
}
#endif // extern "C"

#endif // SPIRV_SPT_H