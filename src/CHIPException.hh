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
 * @file CHIPException.hh
 * @author Paulius Velesko (pvelesko@pglc.io)
 * @brief Header defining CHIP Exceptions
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef CHIP_EXCEPTION_HH
#define CHIP_EXCEPTION_HH

#include "hip/hip_runtime_api.h"
#include "logging.hh"
#include <string>
class CHIPError {
  std::string Msg_;
  hipError_t Err_;

public:
  CHIPError(std::string Msg = "", hipError_t Err = hipErrorUnknown)
      : Msg_(Msg), Err_(Err) {}
  virtual hipError_t toHIPError() { return Err_; }

  std::string getMsgStr() { return Msg_.c_str(); }
  std::string getErrStr() { return std::string(hipGetErrorName(Err_)); }
};

#define CHIPERR_LOG_AND_THROW(msg, errtype)                                    \
  do {                                                                         \
    logError("{} ({}) in {}:{}:{}\n", CHIPError(msg, errtype).getErrStr(),     \
             CHIPError(msg, errtype).getMsgStr(), __FILE__, __LINE__,          \
             __func__);                                                        \
    throw CHIPError(msg, errtype);                                             \
  } while (0)

#define CHIPERR_CHECK_LOG_AND_THROW(status, success, errtype, ...)             \
  do {                                                                         \
    if (status != success) {                                                   \
      std::string error_msg = std::string(resultToString(status));             \
      std::string custom_msg = std::string(__VA_ARGS__);                       \
      std::string msg_ = error_msg + " " + custom_msg;                         \
      CHIPERR_LOG_AND_THROW(msg_, errtype);                                    \
    }                                                                          \
  } while (0)

#define CHIP_TRY try {
#define CHIP_CATCH                                                             \
  }                                                                            \
  catch (CHIPError _status) {                                                  \
    logError("Caught Error: {}", _status.getErrStr());                         \
    Backend->TlsLastError = _status.toHIPError();                              \
    RETURN(_status.toHIPError());                                              \
  }

#define CHIP_CATCH_NO_RETURN                                                   \
  }                                                                            \
  catch (CHIPError _status) {                                                  \
    logError(hipGetErrorName(_status.toHIPError()));                           \
  }

inline void checkIfNullptr(int NumArgs, ...) {
  va_list VaArgList;

  va_start(VaArgList, NumArgs);
  while (NumArgs--)
    if (va_arg(VaArgList, const void *) == nullptr)
      CHIPERR_LOG_AND_THROW("passed in nullptr", hipErrorInvalidHandle);
  va_end(VaArgList);

  return;
}

#define NUMARGS(...) (sizeof((const void *[]){__VA_ARGS__}) / sizeof(void *))
#define NULLCHECK(...) checkIfNullptr(NUMARGS(__VA_ARGS__), __VA_ARGS__);

#define CHECK(X)                                                               \
  if (X == nullptr) {                                                          \
    CHIPERR_LOG_AND_THROW("Nullarg", hipErrorInvalidHandle);                   \
  }

#define CHIPASSERT(X)                                                          \
  do {                                                                         \
    if (!(X)) {                                                                \
      std::string Msg = std::string(__FILE__) + ":";                           \
      Msg += std::to_string(__LINE__) + ": Assertion `";                       \
      Msg += #X;                                                               \
      Msg += "' failed.";                                                      \
      CHIPERR_LOG_AND_THROW(Msg, hipErrorTbd);                                 \
    }                                                                          \
  } while (0)

#endif // ifdef guard
