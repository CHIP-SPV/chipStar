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
#include "CHIPBindingsInternal.hh"
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
  std::string getErrStr() { return std::string(hipGetErrorNameInternal(Err_)); }
};

#define CHIPERR_LOG_AND_THROW(msg, errtype)                                    \
  do {                                                                         \
    logError("{} ({}) in {}:{}:{}\n", CHIPError(msg, errtype).getErrStr(),     \
             CHIPError(msg, errtype).getMsgStr(), __FILE__, __LINE__,          \
             __func__);                                                        \
    throw CHIPError(msg, errtype);                                             \
  } while (0)

#define CHIPERR_LOG_AND_ABORT(msg)                                             \
  do {                                                                         \
    logError("{} in {}:{}:{}\n", msg, __FILE__, __LINE__, __func__);           \
    std::abort();                                                              \
  } while (0)

#define CHIP_TRY try {
#define CHIP_CATCH                                                             \
  }                                                                            \
  catch (CHIPError _status) {                                                  \
    logError("Caught Error: {}", _status.getErrStr());                         \
    RETURN(_status.toHIPError());                                              \
  }

#define CHIP_CATCH_NO_RETURN                                                   \
  }                                                                            \
  catch (CHIPError _status) {                                                  \
    logError(hipGetErrorNameInternal(_status.toHIPError()));                   \
  }

#define CHIP_CATCH_RETURN_CODE(code)                                           \
  }                                                                            \
  catch (CHIPError _status) {                                                  \
    logError("Caught Error: {} Returned: {}", _status.getErrStr(),             \
             hipGetErrorNameInternal(code));                                   \
    RETURN(code);                                                              \
  }

inline void checkIfNullptr(std::string_view File, int Line,
                           std::string_view Function, int NumArgs, ...) {
  va_list VaArgList;

  va_start(VaArgList, NumArgs);
  while (NumArgs--) {
    if (va_arg(VaArgList, const void *) == nullptr) {
      auto Error = CHIPError("passed in nullptr", hipErrorInvalidHandle);
      logError("{} ({}) in {}:{}:{}\n", Error.getErrStr(), Error.getMsgStr(),
               File, Line, Function);
      throw Error;
    }
  }
  va_end(VaArgList);

  return;
}

#define NUMARGS(...) (sizeof((const void *[]){__VA_ARGS__}) / sizeof(void *))
#define NULLCHECK(...)                                                         \
  checkIfNullptr(__FILE__, __LINE__, __func__, NUMARGS(__VA_ARGS__),           \
                 __VA_ARGS__);

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
