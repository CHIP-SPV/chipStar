/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <hip/hip_runtime.h>
#include "test_common.h"

__global__ void tex2DKernel(float *OutputData, hipTextureObject_t TextureObject,
                            int Width, int Height) {
  int X = blockIdx.x * blockDim.x + threadIdx.x;
  int Y = blockIdx.y * blockDim.y + threadIdx.y;
  OutputData[Y * Width + X] = tex2D<float>(TextureObject, X, Y);
}

int runTest(int Argc, char **Argv);

int main(int argc, char **argv) {
  int TestResult = runTest(argc, argv);

  if (TestResult) {
    passed();
  } else {
    exit(EXIT_FAILURE);
  }
}

int runTest(int Argc, char **Argv) {
  int TestResult = 1;
  unsigned int Width = 256;
  unsigned int Height = 256;
  unsigned int Size = Width * Height * sizeof(float);
  float *HData = (float *)malloc(Size);
  memset(HData, 0, Size);
  for (int i = 0; i < Height; i++) {
    for (int j = 0; j < Width; j++) {
      HData[i * Width + j] = i * Width + j;
    }
  }
  printf("hData: ");
  for (int i = 0; i < 64; i++) {
    printf("%f  ", HData[i]);
  }
  printf("\n");

  hipChannelFormatDesc ChannelDesc =
      hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
  hipArray *HipArray;
  hipMallocArray(&HipArray, &ChannelDesc, Width, Height);

  hipMemcpyToArray(HipArray, 0, 0, HData, Size, hipMemcpyHostToDevice);

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
  TexDesc.normalizedCoords = 0;

  // Create texture object
  hipTextureObject_t TextureObject = 0;
  hipCreateTextureObject(&TextureObject, &ResDesc, &TexDesc, NULL);

  float *DData = NULL;
  hipMalloc((void **)&DData, Size);

  dim3 DimBlock(16, 16, 1);
  dim3 DimGrid(Width / DimBlock.x, Height / DimBlock.y, 1);

  hipLaunchKernelGGL(tex2DKernel, dim3(DimGrid), dim3(DimBlock), 0, 0, DData,
                     TextureObject, Width, Height);

  hipDeviceSynchronize();

  float *HOutputData = (float *)malloc(Size);
  memset(HOutputData, 0, Size);
  hipMemcpy(HOutputData, DData, Size, hipMemcpyDeviceToHost);

  printf("dData: ");
  for (int i = 0; i < 64; i++) {
    printf("%f  ", HOutputData[i]);
  }
  printf("\n");
  for (int i = 0; i < Height; i++) {
    for (int j = 0; j < Width; j++) {
      if (HData[i * Width + j] != HOutputData[i * Width + j]) {
        printf("Difference [ %d %d ]:%f ----%f\n", i, j, HData[i * Width + j],
               HOutputData[i * Width + j]);
        TestResult = 0;
        break;
      }
    }
  }
  hipDestroyTextureObject(TextureObject);
  hipFree(DData);
  hipFreeArray(HipArray);
  return TestResult;
}
