/*
 * Copyright (c) 2024 chipStar developers
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

// Test for issue #1062: Global __constant__ memory segfault in hipRTC
// This test reproduces the issue where using global __constant__ memory
// causes a segmentation fault when using hipRTC on Aurora system.

#include "TestCommon.hh"

static constexpr auto Source = R"---(
__constant__ int patchDescriptors = 4;

extern "C"
__global__
void simple(int *y)
{
  y[0] = patchDescriptors;
}
)---";

int main() {
  std::cerr << "Testing global __constant__ memory with hipRTC (Issue #1062)\n";

  // Create program
  auto Prog = HiprtcAssertCreateProgram(Source);

  // Compile program with basic options
  std::vector<const char*> options;
  options.push_back("--std=c++11");
  
  auto Code = HiprtcAssertCompileProgram(Prog, options);

  // Load module
  hipModule_t Module;
  HIP_CHECK(hipModuleLoadData(&Module, Code.data()));

  // Get kernel function
  hipFunction_t Kernel;
  HIP_CHECK(hipModuleGetFunction(&Kernel, Module, "simple"));

  // Allocate device memory
  int *OutputD = nullptr;
  HIP_CHECK(hipMalloc(&OutputD, sizeof(int)));

  // Launch kernel
  void *Args[] = {&OutputD};
  HIP_CHECK(hipModuleLaunchKernel(Kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, Args, nullptr));

  // Copy result back
  int OutputH;
  HIP_CHECK(hipMemcpy(&OutputH, OutputD, sizeof(int), hipMemcpyDeviceToHost));

  // Verify result
  std::cerr << "Result: " << OutputH << " (expected: 4)\n";
  TEST_ASSERT(OutputH == 4);

  // Cleanup
  HIP_CHECK(hipFree(OutputD));
  HIP_CHECK(hipModuleUnload(Module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&Prog));

  std::cerr << "Test passed: Global __constant__ memory works correctly\n";
  return 0;
}
