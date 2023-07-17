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

/**
 * @file CHIPBindingsInternal.hh
 * @author Brice Videau (bvideau@anl.gov)
 * @brief Declaration of internal functions used by the library in serveral
 * locations. These functions exist as a way to avoid reentrancy in our
 * exported symbols.
 * @version 0.1
 * @date 2023-06-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef CHIP_BINDINGS_INTERNAL_H
#define CHIP_BINDINGS_INTERNAL_H
#include "hip/hip_runtime_api.h"

const char *hipGetErrorNameInternal(hipError_t HipError);

hipError_t hipMemcpyFromSymbolAsyncInternal(void *Dst, const void *Symbol,
                                            size_t SizeBytes, size_t Offset,
                                            hipMemcpyKind Kind,
                                            hipStream_t Stream);

hipError_t hipMemcpyToSymbolAsyncInternal(const void *Symbol, const void *Src,
                                          size_t SizeBytes, size_t Offset,
                                          hipMemcpyKind Kind,
                                          hipStream_t Stream);

hipError_t hipEventRecordInternal(hipEvent_t Event, hipStream_t Stream);

hipError_t hipStreamWaitEventInternal(hipStream_t Stream, hipEvent_t Event,
                                      unsigned int Flags);

hipError_t hipMemcpy3DAsyncInternal(const struct hipMemcpy3DParms *Params,
                                    hipStream_t Stream);

hipError_t hipMemcpyInternal(void *Dst, const void *Src, size_t SizeBytes,
                             hipMemcpyKind Kind);
#endif
