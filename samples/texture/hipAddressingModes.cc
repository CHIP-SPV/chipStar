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

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define CHECK_EQUAL(_X, _Y)                                                    \
  do {                                                                         \
    float Expected = _Y;                                                       \
    float Actual = _X;                                                         \
    if (Actual != Expected) {                                                  \
      printf("FAILED: '%s'. Expected '%f'. Got '%f'\n", #_X, Expected,         \
             Actual);                                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void tex2DKernel(float *Output, hipTextureObject_t TexObj, int Width,
                            int Height, float XOffset, float Scale) {
  int X = blockIdx.x * blockDim.x + threadIdx.x;
  int Y = blockIdx.y * blockDim.y + threadIdx.y;
  float Tx = (X + XOffset) * Scale;
  float Ty = Y * Scale;
  Output[Y * Width + X] = tex2D<float>(TexObj, Tx, Ty);
}

hipTextureObject_t createArrayTexture(float *TexData, size_t Width,
                                      size_t Height,
                                      hipTextureAddressMode AddrMode,
                                      bool NormalizedCoords) {

  hipChannelFormatDesc ChannelDesc =
      hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
  hipArray *Array;
  HIPCHECK(hipMallocArray(&Array, &ChannelDesc, Width, Height));
  HIPCHECK(hipMemcpy2DToArray(Array, 0, 0, TexData, Width * sizeof(float),
                              Width * sizeof(float), Height,
                              hipMemcpyHostToDevice));

  hipResourceDesc ResDesc;
  memset(&ResDesc, 0, sizeof(ResDesc));
  ResDesc.resType = hipResourceTypeArray;
  ResDesc.res.array.array = Array;

  // Specify texture object parameters
  hipTextureDesc TexDesc;
  memset(&TexDesc, 0, sizeof(TexDesc));
  TexDesc.addressMode[0] = AddrMode;
  TexDesc.addressMode[1] = AddrMode;
  TexDesc.filterMode = hipFilterModePoint;
  TexDesc.readMode = hipReadModeElementType;
  TexDesc.normalizedCoords = NormalizedCoords;

  // Create texture object
  hipTextureObject_t TexObj = 0;
  HIPCHECK(hipCreateTextureObject(&TexObj, &ResDesc, &TexDesc, NULL));
  return TexObj;
}

void deleteArrayTexture(hipTextureObject_t TexObj) {
  hipResourceDesc ResDesc;
  HIPCHECK(hipGetTextureObjectResourceDesc(&ResDesc, TexObj));
  hipArray *Array = ResDesc.res.array.array;
  HIPCHECK(hipDestroyTextureObject(TexObj));
  HIPCHECK(hipFreeArray(Array));
}

void testAddressMode(hipTextureAddressMode AddrMode, bool NormalizedCoords) {
  const size_t TexWidth = 8;
  const size_t TexHeight = TexWidth;
  const size_t TexSize = TexWidth * TexHeight;
  const size_t GridWidth = TexWidth + 4;
  const size_t GridHeight = TexHeight;
  const size_t GridSize = GridWidth * GridHeight;

  std::vector<float> TexData(TexSize, 0.0f);
  for (size_t I = 0; I < TexSize; I++)
    TexData[I] = I + 1;

  // std::cerr << "Tex:\n";
  // for (size_t J = 0; J < TexHeight; J++) {
  //   for (size_t I = 0; I < TexWidth; I++)
  //     std::cerr << " " << std::setw(2) << TexData[J * TexWidth + I];
  //   std::cerr << "\n";
  // }

  hipTextureObject_t TexObj = createArrayTexture(
      TexData.data(), TexWidth, TexHeight, AddrMode, NormalizedCoords);

  float *DevOutput = NULL;
  HIPCHECK(hipMalloc((void **)&DevOutput, GridSize * sizeof(float)));

  dim3 DimBlock(1, 1, 1);
  dim3 DimGrid(GridWidth, GridHeight, 1);
  hipLaunchKernelGGL(tex2DKernel, dim3(DimGrid), dim3(DimBlock), 0, 0,
                     DevOutput, TexObj, GridWidth, GridHeight, -2.f,
                     (NormalizedCoords ? 1.f / TexWidth : 1.f));
  HIPCHECK(hipDeviceSynchronize());

  std::vector<float> Output(GridSize, -1.0f);
  HIPCHECK(hipMemcpy(Output.data(), DevOutput, GridSize * sizeof(float),
                     hipMemcpyDeviceToHost));

  // std::cerr << "Output:\n";
  // for (size_t J = 0; J < GridHeight; J++) {
  //   for (size_t I = 0; I < GridWidth; I++)
  //     std::cerr << " " << std::setw(2) << Output[J * GridWidth + I];
  //   std::cerr << "\n";
  // }

  // Wrap and mirror modes are not available if normalizedCoords is false.
  // Addr mode falls back to clamp mode.
  if (!NormalizedCoords &&
      (AddrMode == hipAddressModeWrap || AddrMode == hipAddressModeMirror))
    AddrMode = hipAddressModeClamp;

  // Check in-bounds texture reads.
  CHECK_EQUAL(Output[2], 1);
  CHECK_EQUAL(Output[9], 8);

  // Check Out-of-bounds texture reads.
  if (AddrMode == hipAddressModeWrap) {
    CHECK_EQUAL(Output[0], 7);
    CHECK_EQUAL(Output[1], 8);
    CHECK_EQUAL(Output[10], 1);
    CHECK_EQUAL(Output[11], 2);
  } else if (AddrMode == hipAddressModeMirror) {
    CHECK_EQUAL(Output[0], 2);
    CHECK_EQUAL(Output[1], 1);
    CHECK_EQUAL(Output[10], 8);
    CHECK_EQUAL(Output[11], 7);
  } else if (AddrMode == hipAddressModeClamp) {
    CHECK_EQUAL(Output[0], 1);
    CHECK_EQUAL(Output[1], 1);
    CHECK_EQUAL(Output[10], 8);
    CHECK_EQUAL(Output[11], 8);
  } else if (AddrMode == hipAddressModeBorder) {
    CHECK_EQUAL(Output[0], 0);
    CHECK_EQUAL(Output[1], 0);
    CHECK_EQUAL(Output[10], 0);
    CHECK_EQUAL(Output[11], 0);
  } else {
    printf("Unknown address mode!\n");
    exit(1);
  }

  HIPCHECK(hipFree(DevOutput));
  deleteArrayTexture(TexObj);
}

int main(int argc, char **argv) {
  printf("Check hipAddressModeClamp and NormalizedCoords=false:\n");
  testAddressMode(hipAddressModeClamp, false);

  printf("Check hipAddressModeBorder and NormalizedCoords=false:\n");
  testAddressMode(hipAddressModeBorder, false);

  printf("Check hipAddressModeWrap and NormalizedCoords=false:\n");
  testAddressMode(hipAddressModeWrap, false);

  printf("Check hipAddressModeMirror and NormalizedCoords=false:\n");
  testAddressMode(hipAddressModeMirror, false);

  printf("Check hipAddressModeWrap and NormalizedCoords=true:\n");
  testAddressMode(hipAddressModeWrap, true);

  printf("Check hipAddressModeMirror and NormalizedCoords=true:\n");
  testAddressMode(hipAddressModeMirror, true);

  passed();
}
