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

#include "TestCommon.hh"

#include <regex>

static constexpr auto DefaultSource = R"---(
__global__ void add1(int *Dst, const int *Src) { *Dst = *Src + 1; }
)---";

static constexpr auto AssertFooMacro = R"---(
#ifndef FOO
#error FOO was not set!
#endif
__global__ void add1(int *Dst, const int *Src) { *Dst = *Src + 1; }
)---";

static constexpr auto Greet = R"---(
#ifndef GREETING
#error GREETING was not set!
#endif
__global__ void greet(char *Dst) {
  const char *Greeting = GREETING;
  for (size_t i = 0; i < sizeof(GREETING), i++)
    Dst[i] = Greeting[i];
}
)---";

static hiprtcResult compile(hiprtcProgram Program, std::string_view Option,
                            std::string &LogOut) {
  const char *OptArray[] = {Option.data()};
  auto Result = hiprtcCompileProgram(Program, 1, OptArray);

  size_t LogSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(Program, &LogSize));
  if (LogSize) {
    LogOut = std::string(LogSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(Program, &LogOut[0]));
  }

  if (Result != HIPRTC_SUCCESS)
    std::cerr << LogOut;

  return Result;
}

void checkOption(std::string_view Option, const char *Source = DefaultSource) {
  std::cerr << "Checking option '" << Option << "'.\n";
  auto Program = HiprtcAssertCreateProgram(Source);

  std::string Log;
  auto Result = compile(Program, Option, Log);

  TEST_ASSERT(!std::regex_search(Log, std::regex("warning: ignored option")));
  HIPRTC_CHECK(Result);
  HIPRTC_CHECK(hiprtcDestroyProgram(&Program));
}

void checkIgnored(std::string_view Option) {
  std::cerr << "Checking ignored option '" << Option << "'.\n";
  auto Program = HiprtcAssertCreateProgram(DefaultSource);

  std::string Log;
  auto Result = compile(Program, Option, Log);

  TEST_ASSERT(std::regex_search(Log, std::regex("warning: ignored option")));
  HIPRTC_CHECK(Result);
  HIPRTC_CHECK(hiprtcDestroyProgram(&Program));
}

int main() {

  checkOption("--std=c++11");
  checkOption("-O1");
  checkOption("-O2");
  checkOption("-O3");
  checkOption("-DFOO=123", AssertFooMacro);
  // TODO: Does not work with current hipcc probably due to lack of shell
  //       escaping.
  // checkOption("-DGREETING=\"Hello, World!\"", Greet);

  checkIgnored("--nonexistent-flag");
  checkIgnored("non_option");

  return 0;
}
