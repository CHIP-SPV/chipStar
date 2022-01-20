/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hiprtc.h>
#include "macros.hh"
#include "logging.hh"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>

#if !defined(_WIN32)
#pragma GCC visibility push(default)
#endif

const char* hiprtcGetErrorString(hiprtcResult Result) {
  UNIMPLEMENTED(nullptr);
}

hiprtcResult hiprtcVersion(int* Major, int* Minor) {
  UNIMPLEMENTED(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
}

hiprtcResult hiprtcAddNameExpression(hiprtcProgram Prog,
                                     const char* NameExpression) {
  UNIMPLEMENTED(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
}

hiprtcResult hiprtcCompileProgram(hiprtcProgram Prog, int NumOptions,
                                  const char** Options) {
  UNIMPLEMENTED(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
}

hiprtcResult hiprtcCreateProgram(hiprtcProgram* Prog, const char* Src,
                                 const char* Name, int NumHeaders,
                                 const char** Headers,
                                 const char** IncludeNames) {
  UNIMPLEMENTED(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
}

hiprtcResult hiprtcDestroyProgram(hiprtcProgram* Prog) {
  UNIMPLEMENTED(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
}

hiprtcResult hiprtcGetLoweredName(hiprtcProgram Prog,
                                  const char* NameExpression,
                                  const char** LoweredName) {
  UNIMPLEMENTED(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
}

hiprtcResult hiprtcGetProgramLog(hiprtcProgram Prog, char* Log) {
  UNIMPLEMENTED(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
}

hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram Prog, size_t* LogSizeRet) {
  UNIMPLEMENTED(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
}

hiprtcResult hiprtcGetCode(hiprtcProgram Prog, char* Code) {
  UNIMPLEMENTED(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
}

hiprtcResult hiprtcGetCodeSize(hiprtcProgram Prog, size_t* CodeSizeRet) {
  UNIMPLEMENTED(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
}

#if !defined(_WIN32)
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */
