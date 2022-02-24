/*
Copyright (c) 2022 Henry Linjamäki / Parmance for Argonne National Laboratory

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

#include <hip/hip_runtime.h>
#include "test_common.h"

#include <random>

template <typename T>
__global__ void tex2DKernel(T *OutputData, hipTextureObject_t TextureObject,
                            int Width, int Height) {
  int X = blockIdx.x * blockDim.x + threadIdx.x;
  int Y = blockIdx.y * blockDim.y + threadIdx.y;
  if (X < Width && Y < Height)
    OutputData[Y * Width + X] = tex2D<T>(TextureObject, X, Y);
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

template <typename T> bool checkTex2D() {
  constexpr unsigned Width = 13;
  constexpr unsigned Height = 43;
  constexpr unsigned SizeInElts = Width * Height;
  constexpr unsigned WidthInBytes = Width * sizeof(T);
  constexpr unsigned SizeInBytes = SizeInElts * sizeof(T);
  std::vector<T> InputH(SizeInElts);
  for (int i = 0; i < SizeInElts; i++)
    InputH[i] = T(i + 1);

  // std::cout << "InputH:\n";
  // for (const auto &E : InputH)
  //   std::cout << "  " << +E;
  // std::cout << "\n";

  hipChannelFormatDesc ChannelDesc = hipCreateChannelDesc<T>();
  hipArray *InputArray;
  HIPCHECK(hipMallocArray(&InputArray, &ChannelDesc, Width, Height));
  HIPCHECK(hipMemcpy2DToArray(InputArray, 0, 0, InputH.data(), WidthInBytes,
                              WidthInBytes, Height, hipMemcpyHostToDevice));

  hipResourceDesc ResDesc;
  memset(&ResDesc, 0, sizeof(ResDesc));
  ResDesc.resType = hipResourceTypeArray;
  ResDesc.res.array.array = InputArray;

  hipTextureDesc TexDesc;
  memset(&TexDesc, 0, sizeof(TexDesc));
  TexDesc.readMode = hipReadModeElementType;

  // Create texture object
  hipTextureObject_t TextureObject = 0;
  HIPCHECK(hipCreateTextureObject(&TextureObject, &ResDesc, &TexDesc, nullptr));

  T *OutputD;
  HIPCHECK(hipMalloc(&OutputD, SizeInBytes));

  dim3 DimBlock(1, 1, 1);
  dim3 DimGrid(Width, Height, 1);
  hipLaunchKernelGGL(tex2DKernel<T>, dim3(DimGrid), dim3(DimBlock), 0, 0,
                     OutputD, TextureObject, Width, Height);
  HIPCHECK(hipDeviceSynchronize());

  std::vector<T> OutputH(SizeInElts, 0);
  HIPCHECK(
      hipMemcpy(OutputH.data(), OutputD, SizeInBytes, hipMemcpyDeviceToHost));

  // std::cout << "OutputH:\n";
  // for (int i = 0; i < SizeInElts; i++)
  //   std::cout << "  " << +OutputH[i];
  // std::cout << "\n";

  bool TestResult = true;
  for (int i = 0; i < SizeInElts; i++) {
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
  HIPCHECK(hipFreeArray(InputArray));
  return TestResult;
}

int main(int argc, char **argv) {
  bool TestResult = true;
  TestResult &= checkTex2D<char>();
  TestResult &= checkTex2D<unsigned char>();
  TestResult &= checkTex2D<short>();
  TestResult &= checkTex2D<unsigned short>();
  TestResult &= checkTex2D<int>();
  TestResult &= checkTex2D<unsigned int>();
  TestResult &= checkTex2D<float>();

  TestResult &= checkTex2D<char1>();
  TestResult &= checkTex2D<uchar1>();
  TestResult &= checkTex2D<short1>();
  TestResult &= checkTex2D<ushort1>();
  TestResult &= checkTex2D<int1>();
  TestResult &= checkTex2D<uint1>();
  TestResult &= checkTex2D<float1>();

  TestResult &= checkTex2D<char2>();
  TestResult &= checkTex2D<uchar2>();
  TestResult &= checkTex2D<short2>();
  TestResult &= checkTex2D<ushort2>();
  TestResult &= checkTex2D<int2>();
  TestResult &= checkTex2D<uint2>();
  TestResult &= checkTex2D<float2>();

  TestResult &= checkTex2D<char4>();
  TestResult &= checkTex2D<uchar4>();
  TestResult &= checkTex2D<short4>();
  TestResult &= checkTex2D<ushort4>();
  TestResult &= checkTex2D<int4>();
  TestResult &= checkTex2D<uint4>();
  TestResult &= checkTex2D<float4>();

  if (TestResult) {
    passed();
  }
  return 1;
}
