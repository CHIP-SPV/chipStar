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

// The smallest hipRTC kernel that needs fp64, kept as the canary for the gap
// the build-time doubles guard cannot cover.
//
// CHIP_SKIP_TESTS_WITH_DOUBLES wraps every ctest in
// `spirv-extractor --check-for-doubles`, which reads the SPIR-V embedded in the
// test executable and skips the test when it finds a double. A kernel handed to
// hipRTC as a string is compiled after the process starts and lives in no
// embedded module, so that wrapper structurally cannot see it: on a device
// without fp64 the program build fails instead
// (hipErrorSharedObjectInitFailed). Nothing the build system can inspect covers
// that, so a hipRTC test whose kernel uses double has to ask for the skip
// itself.

#include "TestCommon.hh"

static constexpr auto DoubleKernelSource = R"---(
extern "C" __global__ void scale(double *Out) { Out[0] = Out[0] * 2.0; }
)---";

int main() {
  SkipIfDeviceHasNoDoubles();

  double *Out = nullptr;
  HIP_CHECK(hipMalloc(&Out, sizeof(double)));
  double Input = 21.0;
  HIP_CHECK(hipMemcpy(Out, &Input, sizeof(double), hipMemcpyHostToDevice));

  auto Program = HiprtcAssertCreateProgram(DoubleKernelSource);
  auto Code = HiprtcAssertCompileProgram(Program);

  hipModule_t Module;
  hipFunction_t Kernel;
  HIP_CHECK(hipModuleLoadData(&Module, Code.data()));
  HIP_CHECK(hipModuleGetFunction(&Kernel, Module, "scale"));

  void *Args[] = {&Out};
  HIP_CHECK(hipModuleLaunchKernel(Kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, Args,
                                  nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  double Result = 0.0;
  HIP_CHECK(hipMemcpy(&Result, Out, sizeof(double), hipMemcpyDeviceToHost));
  TEST_ASSERT(Result == 42.0);

  HIPRTC_CHECK(hiprtcDestroyProgram(&Program));
  HIP_CHECK(hipModuleUnload(Module));
  HIP_CHECK(hipFree(Out));

  std::cout << "PASSED\n";
  return 0;
}
