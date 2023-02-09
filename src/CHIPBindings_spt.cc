
/*
 * Copyright (c) 2021-23 CHIP-SPV developers
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
 * @file CHIIPBindings_spt.cc
 * @author Paulius Velesko (pvelesko@pglc.io)
 * @brief Implementations of the HIP API functions for explicitly calling the
 * per-thread versions of the functions.
 * @version 0.1
 * @date 2023-02-09
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "hip/spirv_spt.h"
#include "macros.hh"

hipError_t hipMemcpy_spt(void *dst, const void *src, size_t sizeBytes,
                         hipMemcpyKind kind) {
  hipMemcpyAsync(dst, src, sizeBytes, kind, hipStreamPerThread);
  hipStreamSynchronize(hipStreamPerThread);
}

hipError_t hipMemcpyToSymbol_spt(const void *symbol, const void *src,
                                 size_t sizeBytes, size_t offset,
                                 hipMemcpyKind kind) {
  hipMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, kind,
                         hipStreamPerThread);
  hipStreamSynchronize(hipStreamPerThread);
}

hipError_t hipMemcpyFromSymbol_spt(void *dst, const void *symbol,
                                   size_t sizeBytes, size_t offset,
                                   hipMemcpyKind kind) {
  hipMemcpyFromSymbolAsync(dst, symbol, sizeBytes, offset, kind,
                           hipStreamPerThread);
  hipStreamSynchronize(hipStreamPerThread);
}

hipError_t hipMemcpy2D_spt(void *dst, size_t dpitch, const void *src,
                           size_t spitch, size_t width, size_t height,
                           hipMemcpyKind kind) {
  hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind,
                   hipStreamPerThread);
  hipStreamSynchronize(hipStreamPerThread);
}

hipError_t hipMemcpy2DToArray_spt(hipArray *dst, size_t wOffset, size_t hOffset,
                                  const void *src, size_t spitch, size_t width,
                                  size_t height, hipMemcpyKind kind) {
  hipMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height,
                          kind, hipStreamPerThread);
  hipStreamSynchronize(hipStreamPerThread);
}

hipError_t hipMemcpy2DFromArray_spt(void *dst, size_t dpitch,
                                    hipArray_const_t src, size_t wOffset,
                                    size_t hOffset, size_t width, size_t height,
                                    hipMemcpyKind kind) {
  hipMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height,
                            kind, hipStreamPerThread);
  hipStreamSynchronize(hipStreamPerThread);
}

hipError_t hipMemcpy3D_spt(const struct hipMemcpy3DParms *p) {
  hipMemcpy3DAsync(p, hipStreamPerThread);
  hipStreamSynchronize(hipStreamPerThread);
}

hipError_t hipMemset_spt(void *dst, int value, size_t sizeBytes) {
  hipMemsetAsync(dst, value, sizeBytes, hipStreamPerThread);
  hipStreamSynchronize(hipStreamPerThread);
}

hipError_t hipMemset2D_spt(void *dst, size_t pitch, int value, size_t width,
                           size_t height) {
  hipMemset2DAsync(dst, pitch, value, width, height, hipStreamPerThread);
  hipStreamSynchronize(hipStreamPerThread);
}

hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr, int value,
                           hipExtent extent) {
  hipMemset3DAsync(pitchedDevPtr, value, extent, hipStreamPerThread);
  hipStreamSynchronize(hipStreamPerThread);
}

hipError_t hipMemcpyAsync_spt(void *dst, const void *src, size_t sizeBytes,
                              hipMemcpyKind kind, hipStream_t stream) {
  auto Queue = stream ? stream : hipStreamPerThread;
  hipMemcpyAsync(dst, src, sizeBytes, kind, Queue);
}

hipError_t hipStreamQuery_spt(hipStream_t stream) {
  auto Queue = stream ? stream : hipStreamPerThread;
  hipStreamQuery(Queue);
}

hipError_t hipStreamSynchronize_spt(hipStream_t stream) {
  auto Queue = stream ? stream : hipStreamPerThread;
  hipStreamSynchronize(Queue);
}

hipError_t hipStreamGetPriority_spt(hipStream_t stream, int *priority) {
  auto Queue = stream ? stream : hipStreamPerThread;
  hipStreamGetPriority(Queue, priority);
}

hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event,
                                  unsigned int flags) {
  auto Queue = stream ? stream : hipStreamPerThread;
  hipStreamWaitEvent(Queue, event, flags);
}

hipError_t hipEventRecord_spt(hipEvent_t Event, hipStream_t Stream) {
  auto Queue = Stream ? Stream : hipStreamPerThread;
  hipEventRecord(Event, Queue);
}

hipError_t hipStreamGetFlags_spt(hipStream_t stream, unsigned int *flags) {
  auto Queue = stream ? stream : hipStreamPerThread;
  hipStreamGetFlags(Queue, flags);
}

hipError_t hipLaunchCooperativeKernel_spt(const void *f, dim3 gridDim,
                                          dim3 blockDim, void **kernelParams,
                                          uint32_t sharedMemBytes,
                                          hipStream_t hStream) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
#ifdef __cplusplus
extern "C" {
#endif
hipError_t hipLaunchKernel_spt(const void *function_address, dim3 numBlocks,
                               dim3 dimBlocks, void **args,
                               size_t sharedMemBytes, hipStream_t stream) {
  auto Queue = stream ? stream : hipStreamPerThread;
  hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes,
                  Queue);
}
#ifdef __cplusplus
}
#endif // extern "C"
