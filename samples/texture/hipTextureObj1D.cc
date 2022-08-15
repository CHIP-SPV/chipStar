/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 * Copyright (c) 2022 Henry Linjamäki / Parmance for Argonne National Laboratory
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

#include <hip/hip_runtime.h>
#include "test_common.h"

#include <random>

template <typename T>
__global__ void tex1DKernel(T *OutputData, hipTextureObject_t TextureObject,
                            int Width) {
  int X = blockIdx.x * blockDim.x + threadIdx.x;
  if (X < Width)
    OutputData[X] = tex1D<T>(TextureObject, X);
}

template <typename T> std::string getAsString(const T &Val) {
  return std::to_string(Val);
}

#define DEF_GETASSTRING_VEC1(_ELT)                                             \
  template <> std::string getAsString(const _ELT##1 & Val) {                   \
    return "<" + std::to_string(Val.x) + ">";                                  \
  }                                                                            \
  __asm__("")
#define DEF_GETASSTRING_VEC2(_ELT)                                             \
  template <> std::string getAsString(const _ELT##2 & Val) {                   \
    return "<" + std::to_string(Val.x) + ", " + std::to_string(Val.y) + ">";   \
  }                                                                            \
  __asm__("")
#define DEF_GETASSTRING_VEC4(_ELT)                                             \
  template <> std::string getAsString(const _ELT##4 & Val) {                   \
    return "<" + std::to_string(Val.x) + ", " + std::to_string(Val.y) + ", " + \
           std::to_string(Val.z) + ", " + std::to_string(Val.w) + ">";         \
  }                                                                            \
  __asm__("")

#define DEF_GETASSTRING_VECN(_RANK)                                            \
  DEF_GETASSTRING_VEC##_RANK(char);                                            \
  DEF_GETASSTRING_VEC##_RANK(uchar);                                           \
  DEF_GETASSTRING_VEC##_RANK(short);                                           \
  DEF_GETASSTRING_VEC##_RANK(ushort);                                          \
  DEF_GETASSTRING_VEC##_RANK(int);                                             \
  DEF_GETASSTRING_VEC##_RANK(uint);                                            \
  DEF_GETASSTRING_VEC##_RANK(float)

DEF_GETASSTRING_VECN(1);
DEF_GETASSTRING_VECN(2);
DEF_GETASSTRING_VECN(4);

template <typename T> bool checkTex1D() {
  constexpr unsigned int Width = 71;
  unsigned int SizeInBytes = Width * sizeof(T);
  std::vector<T> InputH(Width);
  for (int i = 0; i < Width; i++)
    InputH[i] = T(i + 1);

  // std::cout << "InputH:\n";
  // for (const auto &E : InputH)
  //   std::cout << "  " << +E;
  // std::cout << "\n";

  T *InputD;
  HIPCHECK(hipMalloc(&InputD, SizeInBytes));
  HIPCHECK(
      hipMemcpy(InputD, InputH.data(), SizeInBytes, hipMemcpyHostToDevice));

  hipResourceDesc ResDesc;
  memset(&ResDesc, 0, sizeof(ResDesc));
  ResDesc.resType = hipResourceTypeLinear;
  ResDesc.res.linear.devPtr = InputD;
  ResDesc.res.linear.desc = hipCreateChannelDesc<T>();
  ResDesc.res.linear.sizeInBytes = SizeInBytes;

  hipTextureDesc TexDesc;
  memset(&TexDesc, 0, sizeof(TexDesc));
  TexDesc.readMode = hipReadModeElementType;

  // Create texture object
  hipTextureObject_t TextureObject = 0;
  HIPCHECK(hipCreateTextureObject(&TextureObject, &ResDesc, &TexDesc, nullptr));

  T *OutputD;
  HIPCHECK(hipMalloc(&OutputD, SizeInBytes));

  dim3 DimBlock(1, 1, 1);
  dim3 DimGrid(Width, 1, 1);
  hipLaunchKernelGGL(tex1DKernel<T>, dim3(DimGrid), dim3(DimBlock), 0, 0,
                     OutputD, TextureObject, Width);
  HIPCHECK(hipDeviceSynchronize());

  std::vector<T> OutputH(Width, 0);
  HIPCHECK(
      hipMemcpy(OutputH.data(), OutputD, SizeInBytes, hipMemcpyDeviceToHost));

  // std::cout << "OutputH:\n";
  // for (int i = 0; i < Width * Height; i++)
  //   std::cout << "  " << +OutputH[i];
  // std::cout << "\n";

  bool TestResult = true;
  for (int i = 0; i < Width; i++) {
    T Expected = InputH[i];
    T Actual = OutputH[i];
    if (Actual != Expected) {
      std::cout << "Error at [" << i << "]: Expected '"
                << getAsString<T>(Expected) << "'. Got '"
                << getAsString<T>(Actual) << "'\n";
      TestResult = false;
      break;
    }
  }

  HIPCHECK(hipDestroyTextureObject(TextureObject));
  HIPCHECK(hipFree(OutputD));
  HIPCHECK(hipFree(InputD));
  return TestResult;
}

int main(int argc, char **argv) {
  bool TestResult = true;
  TestResult &= checkTex1D<char>();
  TestResult &= checkTex1D<unsigned char>();
  TestResult &= checkTex1D<short>();
  TestResult &= checkTex1D<unsigned short>();
  TestResult &= checkTex1D<int>();
  TestResult &= checkTex1D<unsigned int>();
  TestResult &= checkTex1D<float>();

  TestResult &= checkTex1D<char1>();
  TestResult &= checkTex1D<uchar1>();
  TestResult &= checkTex1D<short1>();
  TestResult &= checkTex1D<ushort1>();
  TestResult &= checkTex1D<int1>();
  TestResult &= checkTex1D<uint1>();
  TestResult &= checkTex1D<float1>();

  TestResult &= checkTex1D<char2>();
  TestResult &= checkTex1D<uchar2>();
  TestResult &= checkTex1D<short2>();
  TestResult &= checkTex1D<ushort2>();
  TestResult &= checkTex1D<int2>();
  TestResult &= checkTex1D<uint2>();
  TestResult &= checkTex1D<float2>();

  TestResult &= checkTex1D<char4>();
  TestResult &= checkTex1D<uchar4>();
  TestResult &= checkTex1D<short4>();
  TestResult &= checkTex1D<ushort4>();
  TestResult &= checkTex1D<int4>();
  TestResult &= checkTex1D<uint4>();
  TestResult &= checkTex1D<float4>();

  if (TestResult) {
    passed();
  }
  return 1;
}
