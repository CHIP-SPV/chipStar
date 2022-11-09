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

// Test invoking C++/mangled kernels for which hiprtcAddNameExpression() and
// hiprtcGetLoweredName() are needed.

#include "TestCommon.hh"

// Overloaded kernels do not work with the current clang.
#define TEST_OVERLOADED_KERNELS 0

template <typename OutT, typename InT>
static void checkUnaryKernel(hipModule_t Module, const char *KernelName,
                             InT InputH, OutT ExpectedOutput) {
  std::cerr << "Checking Kernel: " << KernelName << "\n";

  hipFunction_t Kernel;
  HIP_CHECK(hipModuleGetFunction(&Kernel, Module, KernelName));

  OutT OutputH;
  InT *InputD = nullptr;
  OutT *OutputD = nullptr;
  HIP_CHECK(hipMalloc(&OutputD, sizeof(OutT)));
  HIP_CHECK(hipMalloc(&InputD, sizeof(InT)));
  HIP_CHECK(hipMemcpy(InputD, &InputH, sizeof(InT), hipMemcpyHostToDevice));

  void *Args[] = {&OutputD, &InputD};
  HIP_CHECK(hipModuleLaunchKernel(Kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, Args,
                                  nullptr));

  HIP_CHECK(hipMemcpy(&OutputH, OutputD, sizeof(OutT), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(InputD));
  HIP_CHECK(hipFree(OutputD));

  TEST_ASSERT(OutputH == ExpectedOutput);
}

static constexpr auto Source = R"---(
#define MY_FLOAT float
typedef int my_int_t;
__global__ void add1(int *Dst, const int *Src) { *Dst = *Src + 1; }
__global__ void mul2(int *Dst, const int *Src) { *Dst = *Src * 2; }
__global__ void mul2(float *Dst, const float *Src) { *Dst = *Src * 2; }
template<class T> __global__ void sub1(T *Dst, const T *Src) {
  *Dst = *Src - 1;
}
template<class DstT, class SrcT>
__global__ void cast(DstT *Dst, const SrcT *Src) { *Dst = *Src; }
namespace foo {
namespace bar {
__global__ void add2(int *Dst, const int *Src) { *Dst = *Src + 1; }
}
}
)---";

int main() {

  auto Prog = HiprtcAssertCreateProgram(Source);

  HIPRTC_CHECK(hiprtcAddNameExpression(Prog, "add1"));

#if TEST_OVERLOADED_KERNELS
  HIPRTC_CHECK(
      hiprtcAddNameExpression(Prog, "(void(*)(int *, const int *))mul2"));
  HIPRTC_CHECK(
      hiprtcAddNameExpression(Prog, "(void(*)(float *, const float *))mul2"));
#endif

  HIPRTC_CHECK(hiprtcAddNameExpression(Prog, "sub1<int>"));
  HIPRTC_CHECK(hiprtcAddNameExpression(Prog, "sub1<float>"));

  // Both of these name expressions are point to the same kernel.
  HIPRTC_CHECK(hiprtcAddNameExpression(Prog, "cast<int, MY_FLOAT>"));
  HIPRTC_CHECK(hiprtcAddNameExpression(Prog, "cast<my_int_t, float>"));

  HIPRTC_CHECK(hiprtcAddNameExpression(Prog, "foo::bar::add2"));

  auto Code = HiprtcAssertCompileProgram(Prog, {"--std=c++11"});

  TEST_ASSERT(hiprtcAddNameExpression(Prog, "after_compilation") ==
              HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION);

  const char *LoweredName = nullptr;
  TEST_ASSERT(hiprtcGetLoweredName(Prog, "does_not_exist", &LoweredName) ==
              HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID);

  const char *LoweredAdd1;
  auto npos = std::string::npos;
  HIPRTC_CHECK(hiprtcGetLoweredName(Prog, "add1", &LoweredAdd1));
  TEST_ASSERT(std::string_view(LoweredAdd1).find("add1") != npos);

#if TEST_OVERLOADED_KERNELS
  const char *LoweredMul2Int, *LoweredMul2Float;
  HIPRTC_CHECK(hiprtcGetLoweredName(Prog, "(void(*)(int *, const int *))mul2",
                                    &LoweredMul2Int));
  HIPRTC_CHECK(hiprtcGetLoweredName(
      Prog, "(void(*)(float *, const float *))mul2", &LoweredMul2Float));
  TEST_ASSERT(std::string_view(LoweredMul2Int).find("mul2") != npos);
  TEST_ASSERT(std::string_view(LoweredMul2Float).find("mul2") != npos);
  TEST_ASSERT(std::string_view(LoweredMul2Int) != LoweredMul2Float);
#endif

  const char *LoweredSub1Int, *LoweredSub1Float;
  HIPRTC_CHECK(hiprtcGetLoweredName(Prog, "sub1<int>", &LoweredSub1Int));
  HIPRTC_CHECK(hiprtcGetLoweredName(Prog, "sub1<float>", &LoweredSub1Float));
  TEST_ASSERT(std::string_view(LoweredSub1Int).find("sub1") != npos);
  TEST_ASSERT(std::string_view(LoweredSub1Float).find("sub1") != npos);
  TEST_ASSERT(std::string_view(LoweredSub1Int) != LoweredSub1Float);

  const char *LoweredCastF2I0, *LoweredCastF2I1;
  HIPRTC_CHECK(
      hiprtcGetLoweredName(Prog, "cast<int, MY_FLOAT>", &LoweredCastF2I0));
  HIPRTC_CHECK(
      hiprtcGetLoweredName(Prog, "cast<my_int_t, float>", &LoweredCastF2I1));
  TEST_ASSERT(std::string_view(LoweredCastF2I0).find("cast") != npos);
  TEST_ASSERT(std::string_view(LoweredCastF2I1).find("cast") != npos);
  // The expressions should point to the same kernel.
  TEST_ASSERT(std::string_view(LoweredCastF2I0) == LoweredCastF2I1);

  const char *LoweredAdd2;
  HIPRTC_CHECK(hiprtcGetLoweredName(Prog, "foo::bar::add2", &LoweredAdd2));
  TEST_ASSERT(std::string_view(LoweredAdd2).find("add2") != npos);

  hipModule_t Module;
  HIP_CHECK(hipModuleLoadData(&Module, Code.data()));

  checkUnaryKernel(Module, LoweredAdd1, 123, 124);
#if TEST_OVERLOADED_KERNELS
  checkUnaryKernel(Module, LoweredMul2Int, 12, 24);
  checkUnaryKernel(Module, LoweredMul2Float, 2.5f, 5.0f);
#endif
  checkUnaryKernel(Module, LoweredSub1Int, 12, 11);
  checkUnaryKernel(Module, LoweredSub1Float, 1.75f, 0.75f);
  checkUnaryKernel(Module, LoweredCastF2I0, 1.23f, 1);

  HIPRTC_CHECK(hiprtcDestroyProgram(&Prog));

  return 0;
}
