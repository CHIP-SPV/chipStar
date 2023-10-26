/*
 * Copyright (c) 2021-22 chipStar developers
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

#ifndef MACROS_HH
#define MACROS_HH

#include "logging.hh"
#include "iostream"
#include "CHIPException.hh"

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define LOCK(x) std::lock_guard<std::mutex> CONCAT(Lock, __LINE__)(x);

#ifdef CHIP_ERROR_IF_NOT_IMPLEMENTED
#define UNIMPLEMENTED(x)                                                       \
  CHIPERR_LOG_AND_THROW("Called a function which is not implemented",          \
                        hipErrorNotSupported);
#else
#define UNIMPLEMENTED(x)                                                       \
  do {                                                                         \
    logWarn("{}: Called a function which is not implemented", __FUNCTION__);   \
    return x;                                                                  \
  } while (0)
#endif

#define RETURN(x)                                                              \
  do {                                                                         \
    hipError_t err = (x);                                                      \
    CHIPTlsLastError = err;                                                    \
    return err;                                                                \
  } while (0)

#define ERROR_IF(cond, err)                                                    \
  if (cond)                                                                    \
    do {                                                                       \
      logError("Error {} at {}:{} code {}", err, __FILE__, __LINE__, #cond);   \
      CHIPTlsLastError = err;                                                  \
      return err;                                                              \
  } while (0)

#define ERROR_CHECK_DEVNUM(device)                                             \
  ERROR_IF(((device < 0) || ((size_t)device >= Backend->getNumDevices())),     \
           hipErrorInvalidDevice)

#define ERROR_CHECK_DEVHANDLE(device)                                          \
  auto I = std::find(Backend->getDevices().begin(),                            \
                     Backend->getDevices().end(), device);                     \
  ERROR_IF(I == Backend->getDevices().end(), hipErrorInvalidDevice)

#endif
