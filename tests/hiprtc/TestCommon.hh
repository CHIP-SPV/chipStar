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

#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

#define TEST_ASSERT(_X)                                                        \
  do {                                                                         \
    if (!(_X)) {                                                               \
      std::cerr << __FILE__ << ":" << __func__ << ":" << __LINE__              \
                << ": Test assertion failed.\n";                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define HIPRTC_CHECK(_RESULT)                                                  \
  do {                                                                         \
    hiprtcResult ResultCode = (_RESULT);                                       \
    if (ResultCode != HIPRTC_SUCCESS) {                                        \
      std::cerr << "Failure at " << __FILE__ << ":" << __func__ << ":"         \
                << __LINE__ << ": " << hiprtcGetErrorString(ResultCode)        \
                << "\n";                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define HIP_CHECK(_RESULT)                                                     \
  do {                                                                         \
    hipError_t ResultCode = (_RESULT);                                         \
    if (ResultCode != hipSuccess) {                                            \
      std::cerr << "Failure at " << __FILE__ << ":" << __func__ << ":"         \
                << __LINE__ << ": " << hipGetErrorString(ResultCode) << "\n";  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

hiprtcProgram HiprtcAssertCreateProgram(const std::string &Src) {
  hiprtcProgram Program;
  HIPRTC_CHECK(
      hiprtcCreateProgram(&Program, Src.c_str(), "foo", 0, nullptr, nullptr));
  return Program;
}

std::vector<char> HiprtcAssertCompileProgram(
    hiprtcProgram Program,
    const std::vector<const char *> &Options = std::vector<const char *>()) {
  auto Result = hiprtcCompileProgram(Program, Options.size(),
                                     (const char **)Options.data());

  size_t LogSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(Program, &LogSize));
  if (Result != HIPRTC_SUCCESS && LogSize) {
    std::string Log(LogSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(Program, &Log[0]));
    std::cerr << Log << "\n";
  }

  HIPRTC_CHECK(Result);

  size_t CodeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(Program, &CodeSize));
  std::vector<char> Code(CodeSize);
  HIPRTC_CHECK(hiprtcGetCode(Program, Code.data()));

  return Code;
}
