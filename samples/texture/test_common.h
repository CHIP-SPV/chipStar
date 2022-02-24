/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * File is intended to C and CPP compliant hence any CPP specic changes
 * should be added into CPP section
 *
 */

#ifdef __cplusplus
#include <iostream>
#include <iomanip>
#if __CUDACC__
#include <sys/time.h>
#else
#include <chrono>
#endif
#endif

// ************************ GCC section **************************
#include <stddef.h>

#include "hip/hip_runtime.h"

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define passed()                                                               \
  printf("%sPASSED!%s\n", KGRN, KNRM);                                         \
  exit(0);

// The real "assert" would have written to stderr. But it is
// sufficient to just fflush here without getting pedantic. This also
// ensures that we don't lose any earlier writes to stdout.
#define failed(...)                                                            \
  printf("%serror: ", KRED);                                                   \
  printf(__VA_ARGS__);                                                         \
  printf("\n");                                                                \
  printf("error: TEST FAILED\n%s", KNRM);                                      \
  fflush(NULL);                                                                \
  abort();

#define warn(...)                                                              \
  printf("%swarn: ", KYEL);                                                    \
  printf(__VA_ARGS__);                                                         \
  printf("\n");                                                                \
  printf("warn: TEST WARNING\n%s", KNRM);

#define HIP_PRINT_STATUS(status)                                               \
  std::cout << hipGetErrorName(status) << " at line: " << __LINE__ << std::endl;

#define HIPCHECK(error)                                                        \
  {                                                                            \
    hipError_t localError = error;                                             \
    if ((localError != hipSuccess) &&                                          \
        (localError != hipErrorPeerAccessAlreadyEnabled)) {                    \
      printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED,                   \
             hipGetErrorString(localError), localError, #error, __FILE__,      \
             __LINE__, KNRM);                                                  \
      failed("API returned error code.");                                      \
    }                                                                          \
  }

#define HIPASSERT(condition)                                                   \
  if (!(condition)) {                                                          \
    failed("%sassertion %s at %s:%d%s \n", KRED, #condition, __FILE__,         \
           __LINE__, KNRM);                                                    \
  }

#define HIPCHECK_API(API_CALL, EXPECTED_ERROR)                                 \
  {                                                                            \
    hipError_t _e = (API_CALL);                                                \
    if (_e != (EXPECTED_ERROR)) {                                              \
      failed(                                                                  \
          "%sAPI '%s' returned %d(%s) but test expected %d(%s) at %s:%d%s \n", \
          KRED, #API_CALL, _e, hipGetErrorName(_e), EXPECTED_ERROR,            \
          hipGetErrorName(EXPECTED_ERROR), __FILE__, __LINE__, KNRM);          \
    }                                                                          \
  }

// ********************* CPP section *********************
#ifdef __cplusplus
namespace HipTest {

template <typename T>
void checkArray(T Input, T Output, size_t Height, size_t Width) {
  for (int i = 0; i < Height; i++) {
    for (int j = 0; j < Width; j++) {
      int Offset = i * Width + j;
      if (Input[Offset] != Output[Offset]) {
        std::cerr << '[' << i << ',' << j << ',' << "]:" << Input[Offset]
                  << "----" << Output[Offset] << "  ";
        failed("mistmatch at:%d %d", i, j);
      }
    }
  }
}
};     // namespace HipTest
#endif //__cplusplus
