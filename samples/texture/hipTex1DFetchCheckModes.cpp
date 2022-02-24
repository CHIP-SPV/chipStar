// Borrowed from HIP-Common.
/*
Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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

/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

#define N 16
#define offset 3
__global__ void tex1dKernel(float *Val, hipTextureObject_t Obj) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N)
    Val[k] = tex1Dfetch<float>(Obj, k + offset);
}

int runTest(hipTextureAddressMode, hipTextureFilterMode);

int main(int argc, char **argv) {
  int TestResult = runTest(hipAddressModeClamp, hipFilterModePoint);
  TestResult = TestResult & runTest(hipAddressModeClamp, hipFilterModeLinear);
  TestResult = TestResult & runTest(hipAddressModeBorder, hipFilterModePoint);
  TestResult = TestResult & runTest(hipAddressModeBorder, hipFilterModeLinear);
  if (TestResult) {
    passed();
  } else {
    exit(EXIT_FAILURE);
  }
}

int runTest(hipTextureAddressMode AddressMode,
            hipTextureFilterMode FilterMode) {

  int TestResult = 1;

  // Allocating the required buffer on gpu device
  float *TexBuf, *TexBufOut;
  float Val[N], Output[N];

  for (int i = 0; i < N; i++) {
    Val[i] = i + 1;
    Output[i] = 0.0;
  }

  HIPCHECK(hipMalloc(&TexBuf, N * sizeof(float)));
  HIPCHECK(hipMalloc(&TexBufOut, N * sizeof(float)));
  HIPCHECK(hipMemcpy(TexBuf, Val, N * sizeof(float), hipMemcpyHostToDevice));
  HIPCHECK(hipMemset(TexBufOut, 0, N * sizeof(float)));
  hipResourceDesc ResDescLinear;

  memset(&ResDescLinear, 0, sizeof(ResDescLinear));
  ResDescLinear.resType = hipResourceTypeLinear;
  ResDescLinear.res.linear.devPtr = TexBuf;
  ResDescLinear.res.linear.desc =
      hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
  ResDescLinear.res.linear.sizeInBytes = N * sizeof(float);

  hipTextureDesc TexDesc;
  memset(&TexDesc, 0, sizeof(TexDesc));
  TexDesc.readMode = hipReadModeElementType;

  TexDesc.addressMode[0] = AddressMode;
  TexDesc.filterMode = FilterMode;
  TexDesc.normalizedCoords = false;

  // Creating texture object
  hipTextureObject_t TexObj = 0;
  HIPCHECK(hipCreateTextureObject(&TexObj, &ResDescLinear, &TexDesc, NULL));

  dim3 DimBlock(1, 1, 1);
  dim3 DimGrid(N, 1, 1);

  hipLaunchKernelGGL(tex1dKernel, dim3(DimGrid), dim3(DimBlock), 0, 0,
                     TexBufOut, TexObj);
  HIPCHECK(hipDeviceSynchronize());

  HIPCHECK(
      hipMemcpy(Output, TexBufOut, N * sizeof(float), hipMemcpyDeviceToHost));

  for (int i = offset; i < N; i++) {
    if (Output[i - offset] != Val[i]) {
      TestResult = 0;
      break;
    }
  }
  // For hipResourceTypeLinear, reading of out-of-boundary address is undefined!
  // So we won't verify those data

  HIPCHECK(hipDestroyTextureObject(TexObj));
  HIPCHECK(hipFree(TexBuf));
  HIPCHECK(hipFree(TexBufOut));
  printf("%s(addressMode %d, filterMode %d) %s\n", __FUNCTION__, AddressMode,
         FilterMode, TestResult ? "succeed" : "failed");
  return TestResult;
}
