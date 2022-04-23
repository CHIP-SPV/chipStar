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

std::default_random_engine RandomEngine;
template <typename T> static inline T getRandom() {
  auto Signed = std::is_signed<T>::value;
  auto Max = std::numeric_limits<T>::max();
  std::uniform_int_distribution<T> Distrib(Signed ? -Max : 0, Max);
  return Distrib(RandomEngine);
}

template <typename T> int checkNormalizedFloat() {
  constexpr unsigned int Width = 8;
  constexpr unsigned int Height = 8;
  constexpr unsigned WidthInBytes = Width * sizeof(T);
  unsigned int OutputSize = Width * Height * sizeof(float);
  std::vector<T> InputH(Width * Height, 0);
  for (int i = 0; i < Height; i++) {
    for (int j = 0; j < Width; j++) {
      InputH[i * Width + j] = getRandom<T>();
    }
  }
  InputH[0] = std::is_signed<T>::value ? -std::numeric_limits<T>::max() : 0;
  InputH[1] = 0.f;
  InputH[2] = std::numeric_limits<T>::max();

  // std::cout << "InputH:\n";
  // for (const auto &E : InputH)
  //   std::cout << "  " << E;
  // std::cout << "\n";

  std::vector<float> RefData(Width * Height, 0.f);
  for (unsigned i = 0; i < RefData.size(); i++) {
    float Ref =
        std::max<float>(float(InputH[i]) / (std::numeric_limits<T>::max()),
                        std::is_signed<T>::value ? -1.f : 0.f);
    RefData[i] = Ref;
  }

  // std::cout << "RefData:\n";
  // for (const auto &E : RefData)
  //   std::cout << "  " << E;
  // std::cout << "\n";

  hipChannelFormatDesc ChannelDesc = hipCreateChannelDesc<T>();
  hipArray *HipArray;
  HIPCHECK(hipMallocArray(&HipArray, &ChannelDesc, Width, Height));
  HIPCHECK(hipMemcpy2DToArray(HipArray, 0, 0, InputH.data(), WidthInBytes,
                              WidthInBytes, Height, hipMemcpyHostToDevice));

  hipResourceDesc ResDesc;
  memset(&ResDesc, 0, sizeof(ResDesc));
  ResDesc.resType = hipResourceTypeArray;
  ResDesc.res.array.array = HipArray;

  // Specify texture object parameters
  hipTextureDesc TexDesc;
  memset(&TexDesc, 0, sizeof(TexDesc));
  TexDesc.addressMode[0] = hipAddressModeWrap;
  TexDesc.addressMode[1] = hipAddressModeWrap;
  TexDesc.filterMode = hipFilterModePoint;
  TexDesc.readMode = hipReadModeNormalizedFloat;
  TexDesc.normalizedCoords = 0;

  // Create texture object
  hipTextureObject_t TextureObject = 0;
  HIPCHECK(hipCreateTextureObject(&TextureObject, &ResDesc, &TexDesc, NULL));

  float *OutputD = NULL;
  HIPCHECK(hipMalloc((void **)&OutputD, OutputSize));

  dim3 DimBlock(1, 1, 1);
  dim3 DimGrid(Width, Height, 1);
  hipLaunchKernelGGL(tex2DKernel<float>, dim3(DimGrid), dim3(DimBlock), 0, 0,
                     OutputD, TextureObject, Width, Height);
  HIPCHECK(hipDeviceSynchronize());

  std::vector<float> OutputH(Width * Height, 0.f);
  HIPCHECK(
      hipMemcpy(OutputH.data(), OutputD, OutputSize, hipMemcpyDeviceToHost));

  // std::cout << "OutputH:\n";
  // for (int i = 0; i < Width * Height; i++)
  //   std::cout << "  " << OutputH[i];
  // std::cout << "\n";

  int TestResult = 1;
  for (int i = 0; i < Height; i++) {
    for (int j = 0; j < Width; j++) {
      float Expected = RefData[i * Width + j];
      float Actual = OutputH[i * Width + j];
      if (Actual != Expected) {
        std::cout << "Error at [" << i << "][" << j << "]: Expected '"
                  << Expected << "'. Got '" << Actual << "'\n";
        TestResult = 0;
        break;
      }
    }
  }

  HIPCHECK(hipDestroyTextureObject(TextureObject));
  HIPCHECK(hipFree(OutputD));
  HIPCHECK(hipFreeArray(HipArray));
  return TestResult;
}

// Template for checking texel types to which
// the hipReadModeNormalizedFloat setting is not applied.
template <typename T> int checkIgnoreNormalizedFloat() {
  constexpr unsigned Width = 8;
  constexpr unsigned Height = 8;
  constexpr unsigned WidthInBytes = Width * sizeof(T);
  unsigned int OutputSize = Width * Height * sizeof(T);
  std::vector<T> InputH(Width * Height, 0);
  for (int i = 0; i < Height; i++) {
    for (int j = 0; j < Width; j++) {
      InputH[i * Width + j] = getRandom<T>();
    }
  }

  // std::cout << "InputH:\n";
  // for (const auto &E : InputH)
  //   std::cout << "  " << E;
  // std::cout << "\n";

  hipChannelFormatDesc ChannelDesc = hipCreateChannelDesc<T>();
  hipArray *HipArray;
  HIPCHECK(hipMallocArray(&HipArray, &ChannelDesc, Width, Height));
  HIPCHECK(hipMemcpy2DToArray(HipArray, 0, 0, InputH.data(), WidthInBytes,
                              WidthInBytes, Height, hipMemcpyHostToDevice));

  hipResourceDesc ResDesc;
  memset(&ResDesc, 0, sizeof(ResDesc));
  ResDesc.resType = hipResourceTypeArray;
  ResDesc.res.array.array = HipArray;

  // Specify texture object parameters
  hipTextureDesc TexDesc;
  memset(&TexDesc, 0, sizeof(TexDesc));
  TexDesc.addressMode[0] = hipAddressModeWrap;
  TexDesc.addressMode[1] = hipAddressModeWrap;
  TexDesc.filterMode = hipFilterModePoint;
  TexDesc.readMode = hipReadModeNormalizedFloat;
  TexDesc.normalizedCoords = 0;

  // Create texture object
  hipTextureObject_t TextureObject = 0;
  HIPCHECK(hipCreateTextureObject(&TextureObject, &ResDesc, &TexDesc, NULL));

  T *OutputD = NULL;
  HIPCHECK(hipMalloc((void **)&OutputD, OutputSize));

  dim3 DimBlock(1, 1, 1);
  dim3 DimGrid(Width, Height, 1);
  hipLaunchKernelGGL(tex2DKernel<T>, dim3(DimGrid), dim3(DimBlock), 0, 0,
                     OutputD, TextureObject, Width, Height);
  HIPCHECK(hipDeviceSynchronize());

  std::vector<T> OutputH(Width * Height, 0);
  HIPCHECK(
      hipMemcpy(OutputH.data(), OutputD, OutputSize, hipMemcpyDeviceToHost));

  // std::cout << "OutputH:\n";
  // for (int i = 0; i < Width * Height; i++)
  //   std::cout << "  " << OutputH[i];
  // std::cout << "\n";

  int TestResult = 1;
  for (int i = 0; i < Height; i++) {
    for (int j = 0; j < Width; j++) {
      T Expected = InputH[i * Width + j];
      T Actual = OutputH[i * Width + j];
      if (Actual != Expected) {
        std::cout << "Error at [" << i << "][" << j << "]: Expected '"
                  << Expected << "'. Got '" << Actual << "'\n";
        break;
      }
    }
  }

  HIPCHECK(hipDestroyTextureObject(TextureObject));
  HIPCHECK(hipFree(OutputD));
  HIPCHECK(hipFreeArray(HipArray));
  return TestResult;
}

int main(int argc, char **argv) {
  int TestResult = true;
  TestResult &= checkNormalizedFloat<char>();
  TestResult &= checkNormalizedFloat<unsigned char>();
  TestResult &= checkNormalizedFloat<short>();
  TestResult &= checkNormalizedFloat<unsigned short>();

  // hipReadModeNormalizedFloat only applies to 8- and 16-bit integer textures.
  TestResult &= checkIgnoreNormalizedFloat<int>();
  TestResult &= checkIgnoreNormalizedFloat<unsigned int>();

  if (TestResult) {
    passed();
  }
  return 1;
}
