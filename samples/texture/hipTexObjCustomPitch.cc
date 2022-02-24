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
#include "test_common.h"

#define SIZE_H 4
#define SIZE_W 58

__global__ void texture2dCopyKernel(hipTextureObject_t TexObj, float *Dst) {
  for (int i = 0; i < SIZE_H; i++)
    for (int j = 0; j < SIZE_W; j++)
      Dst[SIZE_W * i + j] = tex2D<float>(TexObj, j, i);
}

static size_t roundUp(size_t X, size_t Align) {
  return ((X + Align - 1) / Align) * Align;
}

void texture2Dtest() {
  float *DevPtrB;
  float *DevPtrA;

  int DeviceID;
  HIPCHECK(hipGetDevice(&DeviceID));

  hipDeviceProp_t DeviceProps;
  HIPCHECK(hipGetDeviceProperties(&DeviceProps, DeviceID));

  if (DeviceProps.texturePitchAlignment < 1) {
    failed("Unsound pitch alignment value '(%lu)'\n",
           DeviceProps.texturePitchAlignment);
  }

  // Pick some valid pitch value > SIZE_W expressed in bytes.
  size_t PitchInBytes =
      roundUp((SIZE_W + 3) * sizeof(float), DeviceProps.texturePitchAlignment);
  size_t PitchInElts = PitchInBytes / sizeof(float);
  float *B = new float[SIZE_H * SIZE_W];
  float *A = new float[SIZE_H * PitchInElts];
  for (unsigned Y = 0; Y < SIZE_H; Y++)
    for (unsigned X = 0; X < PitchInElts; X++) {
      auto Idx = Y * PitchInElts + X;
      A[Idx] = X < SIZE_W ? float(Idx + 1) : -1.f;
    }

  // printf("A:\n");
  // for (unsigned y = 0; y < SIZE_H; y++) {
  //   for (unsigned x = 0; x < PitchInElts; x++) {
  //     printf(" %2.0f", A[y * PitchInElts + x]);
  //   }
  //   printf("\n");
  // }

  HIPCHECK(hipMalloc((void **)&DevPtrA, SIZE_H * PitchInBytes * sizeof(float)));
  HIPCHECK(hipMemcpy(DevPtrA, A, SIZE_H * PitchInBytes * sizeof(float),
                     hipMemcpyHostToDevice));

  // Use the texture object
  hipResourceDesc TexRes;
  memset(&TexRes, 0, sizeof(TexRes));
  TexRes.resType = hipResourceTypePitch2D;
  TexRes.res.pitch2D.devPtr = DevPtrA;
  TexRes.res.pitch2D.height = SIZE_H;
  TexRes.res.pitch2D.width = SIZE_W;
  TexRes.res.pitch2D.pitchInBytes = PitchInBytes;
  TexRes.res.pitch2D.desc = hipCreateChannelDesc<float>();

  hipTextureDesc TexDescr;
  memset(&TexDescr, 0, sizeof(TexDescr));
  TexDescr.normalizedCoords = false;
  TexDescr.filterMode = hipFilterModePoint;
  TexDescr.mipmapFilterMode = hipFilterModePoint;
  TexDescr.addressMode[0] = hipAddressModeClamp;
  TexDescr.addressMode[1] = hipAddressModeClamp;
  TexDescr.readMode = hipReadModeElementType;

  hipTextureObject_t TexObj;
  HIPCHECK(hipCreateTextureObject(&TexObj, &TexRes, &TexDescr, NULL));

  HIPCHECK(hipMalloc((void **)&DevPtrB, SIZE_W * sizeof(float) * SIZE_H));

  hipLaunchKernelGGL(texture2dCopyKernel, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     TexObj, DevPtrB);

  HIPCHECK(hipMemcpy2D(B, SIZE_W * sizeof(float), DevPtrB,
                       SIZE_W * sizeof(float), SIZE_W * sizeof(float), SIZE_H,
                       hipMemcpyDeviceToHost));

  // printf("B:\n");
  // for (unsigned y = 0; y < SIZE_H; y++) {
  //   for (unsigned x = 0; x < SIZE_W; x++) {
  //     printf(" %2.0f", B[y * SIZE_W + x]);
  //   }
  //   printf("\n");
  // }

  for (unsigned Y = 0; Y < SIZE_H; Y++)
    for (unsigned X = 0; X < SIZE_W; X++) {
      float Expected = A[Y * PitchInElts + X];
      float Actual = B[Y * SIZE_W + X];
      if (Expected != Actual) {
        failed("Fail at [%u][%u]: Expected '%f'. Got '%f'.\n", Y, X, Expected,
               Actual);
      }
    }

  delete[] A;
  delete[] B;
  HIPCHECK(hipFree(DevPtrA));
  HIPCHECK(hipFree(DevPtrB));
}

int main() {
  texture2Dtest();
  passed();
}
