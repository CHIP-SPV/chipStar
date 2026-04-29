/*
 * Copyright (c) 2026 chipStar developers
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

// Regression test for the HIPRTC cache + name-expression interaction.
//
// Before the fix that landed alongside this test, hiprtcCompileProgram() on a
// cache hit returned HIPRTC_SUCCESS without populating the name-expression ->
// lowered-name map. hiprtcGetLoweredName() then returned a pointer to an
// empty string, and the caller's downstream hipModuleGetFunction("") failed
// with "Failed to find kernel via kernel name: ".
//
// CHIP_MODULE_CACHE_DIR must be set and non-empty in the environment when
// this test runs (it is read by libCHIP at static-init time, before main()).
// Test running infrastructure (CMakeLists.txt) sets the env var; if you run
// the binary by hand, export CHIP_MODULE_CACHE_DIR=<some dir> first.

#include "TestCommon.hh"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

static constexpr auto Source = R"---(
__global__ void demo_kernel(int *data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = data[i] + 1;
}
)---";

// Compile once and return the (mangled) lowered name for "demo_kernel".
static std::string compileAndGetLoweredName() {
  hiprtcProgram Prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&Prog, Source, "demo.hip", 0, nullptr,
                                   nullptr));
  HIPRTC_CHECK(hiprtcAddNameExpression(Prog, "demo_kernel"));

  hiprtcResult R = hiprtcCompileProgram(Prog, 0, nullptr);
  if (R != HIPRTC_SUCCESS) {
    size_t LogSize;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(Prog, &LogSize));
    if (LogSize) {
      std::string Log(LogSize, '\0');
      hiprtcGetProgramLog(Prog, Log.data());
      std::cerr << Log << "\n";
    }
    HIPRTC_CHECK(R);
  }

  // The bug surfaces here on cache hit: pre-fix, *Mangled is the empty string.
  const char *Mangled = nullptr;
  HIPRTC_CHECK(hiprtcGetLoweredName(Prog, "demo_kernel", &Mangled));
  TEST_ASSERT(Mangled != nullptr);
  std::string Result(Mangled);
  HIPRTC_CHECK(hiprtcDestroyProgram(&Prog));
  return Result;
}

int main() {
  // The cache-hit path can only be exercised when CHIP_MODULE_CACHE_DIR is
  // set. libCHIP reads this env var at static-init time, so we validate it
  // here rather than try to set it ourselves (too late from main).
  const char *CacheDir = std::getenv("CHIP_MODULE_CACHE_DIR");
  if (!CacheDir || !*CacheDir) {
    std::cerr << "CHIP_MODULE_CACHE_DIR is not set in the environment; this "
                 "test is meaningless without it. The CMake build sets it "
                 "for ctest invocations.\n";
    return 1;
  }

  // Wipe any stale 'hiprtc/' subdir from a prior ctest run so Run 1 below is
  // guaranteed to be a cache miss and Run 2 a cache hit.
  std::error_code Ec;
  std::filesystem::remove_all(std::string(CacheDir) + "/hiprtc", Ec);

  // First compile: cache miss (cache dir was just cleared).
  std::string Mangled1 = compileAndGetLoweredName();
  TEST_ASSERT(!Mangled1.empty());
  std::cerr << "Run 1 lowered name: '" << Mangled1 << "'\n";

  // Second compile: cache hit guaranteed (Run 1 just populated it).
  std::string Mangled2 = compileAndGetLoweredName();
  std::cerr << "Run 2 lowered name: '" << Mangled2 << "'\n";

  // The regression assertion: pre-fix, this would be "".
  TEST_ASSERT(!Mangled2.empty());
  TEST_ASSERT(Mangled1 == Mangled2);

  return 0;
}
