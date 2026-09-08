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

#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
// TODO: Remove this header when it is included by the
//       hip_runtime_api.h header.
#include <hip/spirv_hip_host_defines.h>

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

/// Reports HIP_SKIP_THIS_TEST and exits when the device has no fp64.
///
/// A kernel compiled at runtime through hipRTC is in none of the SPIR-V modules
/// embedded in the test executable, which is all that
/// `spirv-extractor --check-for-doubles` can inspect. That is the wrapper every
/// add_hip_test registration goes through when CHIP_SKIP_TESTS_WITH_DOUBLES is
/// on, so it cannot skip a hipRTC test whose kernel uses double, and the device
/// compiler is left to reject the kernel: the module build fails with
/// hipErrorSharedObjectInitFailed on OpenCL and hipErrorInvalidImage on Level
/// Zero, unless IGC is emulating fp64 (IGC_EnableDPEmulation), which
/// arch.hasDoubles does not report. Such a test asks for the skip itself by
/// calling this before it compiles the kernel.
void SkipIfDeviceHasNoDoubles() {
  hipDeviceProp_t Props;
  HIP_CHECK(hipGetDeviceProperties(&Props, 0));
  if (Props.arch.hasDoubles)
    return;
  std::cout << "HIP_SKIP_THIS_TEST: device has no fp64 and the kernel this "
               "test compiles at runtime uses double\n";
  exit(0);
}

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
