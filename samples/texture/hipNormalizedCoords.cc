/*
 * Copyright (c) 2021-22 chipStar developers
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>

__global__ void tex2DKernel(float *OutputData, hipTextureObject_t TextureObject,
                            int Width, int Height) {
  int X = blockIdx.x * blockDim.x + threadIdx.x;
  int Y = blockIdx.y * blockDim.y + threadIdx.y;
  float Tx = float(X) / float(Width);
  float Ty = float(Y) / float(Height);
  OutputData[Y * Width + X] = tex2D<float>(TextureObject, Tx, Ty);
}

int runTest() {
  constexpr unsigned Width = 64;
  constexpr unsigned Height = 128;
  constexpr unsigned WidthInBytes = Width * sizeof(float);
  constexpr unsigned SizeInElts = Width * Height;
  constexpr unsigned SizeInBytes = SizeInElts * sizeof(float);

  std::vector<float> InputH(SizeInElts, 0);
  for (int i = 0; i < SizeInElts; i++)
    InputH[i] = i + 1;

  // printf("InputH: ");
  // for (int i = 0; i < Width; i++)
  //   printf("%f  ", InputH[i]);
  // printf("\n");

  hipChannelFormatDesc ChannelDesc =
      hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
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
  TexDesc.readMode = hipReadModeElementType;
  TexDesc.normalizedCoords = 1;

  // Create texture object
  hipTextureObject_t TextureObject = 0;
  HIPCHECK(hipCreateTextureObject(&TextureObject, &ResDesc, &TexDesc, NULL));

  float *OutputD = NULL;
  HIPCHECK(hipMalloc((void **)&OutputD, SizeInBytes));

  dim3 DimBlock(16, 16, 1);
  dim3 DimGrid(Width / DimBlock.x, Height / DimBlock.y, 1);

  hipLaunchKernelGGL(tex2DKernel, dim3(DimGrid), dim3(DimBlock), 0, 0, OutputD,
                     TextureObject, Width, Height);

  HIPCHECK(hipDeviceSynchronize());

  std::vector<float> OutputH(SizeInElts, 0);
  HIPCHECK(
      hipMemcpy(OutputH.data(), OutputD, SizeInBytes, hipMemcpyDeviceToHost));

  // printf("OutputD: ");
  // for (int i = 0; i < 16; i++) {
  //   printf("%f  ", OutputH[i]);
  // }
  // printf("\n");

  int TestResult = 1;
  for (int i = 0; i < Height; i++) {
    for (int j = 0; j < Width; j++) {
      auto Expected = InputH[i * Width + j];
      auto Actual = OutputH[i * Width + j];
      if (Expected != Actual) {
        printf("Error at [%d][%d]: Expected '%f'. Got '%f'\n", i, j, Expected,
               Actual);
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

int main(int argc, char **argv) {
  int TestResult = runTest();
  if (TestResult) {
    passed();
  }
  return EXIT_FAILURE;
}
