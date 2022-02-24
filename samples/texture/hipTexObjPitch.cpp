/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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

/*HIT_START
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */
#include "test_common.h"

#define SIZE_H 20
#define SIZE_W 179
// texture object is a kernel argument
template <typename TYPE_t>
__global__ void texture2dCopyKernel(hipTextureObject_t TexObj, TYPE_t *Dst,
                                    TYPE_t *A) {

  for (int i = 0; i < SIZE_H; i++)
    for (int j = 0; j < SIZE_W; j++)
      Dst[SIZE_W * i + j] = tex2D<TYPE_t>(TexObj, j, i);
  __syncthreads();
}

template <typename TYPE_t> void texture2Dtest() {
  TYPE_t *B;
  TYPE_t *A;
  TYPE_t *DevPtrB;
  TYPE_t *DevPtrA;

  B = new TYPE_t[SIZE_H * SIZE_W];
  A = new TYPE_t[SIZE_H * SIZE_W];
  for (size_t i = 1; i <= (SIZE_H * SIZE_W); i++) {
    A[i - 1] = i;
  }

  size_t DevPitchA;
  HIPCHECK(hipMallocPitch((void **)&DevPtrA, &DevPitchA,
                          SIZE_W * sizeof(TYPE_t), SIZE_H));
  HIPCHECK(hipMemcpy2D(DevPtrA, DevPitchA, A, SIZE_W * sizeof(TYPE_t),
                       SIZE_W * sizeof(TYPE_t), SIZE_H, hipMemcpyHostToDevice));

  // Use the texture object
  hipResourceDesc TexRes;
  memset(&TexRes, 0, sizeof(TexRes));
  TexRes.resType = hipResourceTypePitch2D;
  TexRes.res.pitch2D.devPtr = DevPtrA;
  TexRes.res.pitch2D.height = SIZE_H;
  TexRes.res.pitch2D.width = SIZE_W;
  TexRes.res.pitch2D.pitchInBytes = DevPitchA;
  TexRes.res.pitch2D.desc = hipCreateChannelDesc<TYPE_t>();

  hipTextureDesc TexDescr;
  memset(&TexDescr, 0, sizeof(TexDescr));
  TexDescr.normalizedCoords = false;
  TexDescr.filterMode = hipFilterModePoint;
  TexDescr.mipmapFilterMode = hipFilterModePoint;
  TexDescr.addressMode[0] = hipAddressModeClamp;
  TexDescr.addressMode[1] = hipAddressModeClamp;
  TexDescr.addressMode[2] = hipAddressModeClamp;
  TexDescr.readMode = hipReadModeElementType;

  hipTextureObject_t TexObj;
  HIPCHECK(hipCreateTextureObject(&TexObj, &TexRes, &TexDescr, NULL));

  HIPCHECK(hipMalloc((void **)&DevPtrB, SIZE_W * sizeof(TYPE_t) * SIZE_H));

  hipLaunchKernelGGL(texture2dCopyKernel, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     TexObj, DevPtrB, DevPtrA);

  HIPCHECK(hipMemcpy2D(B, SIZE_W * sizeof(TYPE_t), DevPtrB,
                       SIZE_W * sizeof(TYPE_t), SIZE_W * sizeof(TYPE_t), SIZE_H,
                       hipMemcpyDeviceToHost));

  HipTest::checkArray(A, B, SIZE_H, SIZE_W);
  delete[] A;
  delete[] B;
  HIPCHECK(hipFree(DevPtrA));
  HIPCHECK(hipFree(DevPtrB));
}

int main() {
  texture2Dtest<float>();
  texture2Dtest<int>();
  texture2Dtest<unsigned char>();
  texture2Dtest<short>();
  texture2Dtest<char>();
  texture2Dtest<unsigned int>();
  passed();
}
