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

#include "TestCommon.hh"

static constexpr auto SourceUsingMacro = R"---(
struct AlignedStruct {
  char data[16];
};

template<int Bytes>
struct AlignTest {
  ALIGN_STRUCT_MACRO(Bytes, AlignedStruct) s;
};

__global__ void testAlignment(int *result) {
  AlignTest<32> test;
  *result = alignof(decltype(test.s));
}
)---";

static constexpr auto SourceWithOptionalDefine = R"---(
#ifndef OPTIONAL_VALUE
#define OPTIONAL_VALUE 42
#endif

__global__ void testOptional(int *result) {
  *result = OPTIONAL_VALUE;
}
)---";

int main() {
  std::cerr << "Test 1: -D option with parentheses (combined argument)\n";
  {
    auto Program = HiprtcAssertCreateProgram(SourceUsingMacro);

    std::vector<const char *> Options = {
      "-DALIGN_STRUCT_MACRO(bytes,struct_defn)=struct_defn __attribute__((aligned(bytes)))"
    };

    auto Code = HiprtcAssertCompileProgram(Program, Options);
    
    HIPRTC_CHECK(hiprtcDestroyProgram(&Program));

    std::cerr << "Test 1 PASSED\n";
  }

  std::cerr << "\nTest 2: Option with parentheses that's NOT a -D flag\n";
  {
    auto Program = HiprtcAssertCreateProgram(SourceWithOptionalDefine);

    // -I path with parentheses - this is a realistic case mentioned by Noerr
    // The paren will cause shell syntax error since it's NOT escaped
    std::vector<const char *> Options = {
      "-I/some/path/(version-1.0)/include",  // NOT a -D option, has parentheses
      "-DOPTIONAL_VALUE=42"
    };

    auto Code = HiprtcAssertCompileProgram(Program, Options);
    
    HIPRTC_CHECK(hiprtcDestroyProgram(&Program));

    std::cerr << "Test 2 PASSED\n";
  }

  std::cerr << "\nTest 3: Filename with shell metacharacters\n";
  {
    auto Program = HiprtcAssertCreateProgram(SourceWithOptionalDefine);

    // Using -include with a filename that has parentheses
    // This should fail with current implementation
    std::vector<const char *> Options = {
      "-DOPTIONAL_VALUE=(1+2)*3",  // This gets escaped (starts with -D)
      "-Wno-unused(test)",  // This does NOT get escaped - contains parentheses
    };

    auto Code = HiprtcAssertCompileProgram(Program, Options);
    
    HIPRTC_CHECK(hiprtcDestroyProgram(&Program));

    std::cerr << "Test 3 PASSED\n";
  }

  std::cerr << "\nAll shell metacharacter escaping tests passed!\n";

  return 0;
}


