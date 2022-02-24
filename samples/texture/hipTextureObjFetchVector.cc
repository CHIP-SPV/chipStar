// Borrowed from upstream HIP-Common
/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hip/hip_runtime.h"

#include <vector>
#include <iostream>

#define HIP_CHECK(x) assert(x == hipSuccess)

template <typename T>
__global__ void tex1dKernelFetch(T *Val, hipTextureObject_t Obj, int N) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    Val[k] = tex1Dfetch<T>(Obj, k);
  }
}

template <typename T> static inline __host__ __device__ constexpr int rank() {
  return sizeof(T) / sizeof(decltype(T::x));
}

template <typename T> static inline T getRandom() {
  double R = 0;
  if (std::is_signed<T>::value) {
    R = (std::rand() - RAND_MAX / 2.0) / (RAND_MAX / 2.0 + 1.);
  } else {
    R = std::rand() / (RAND_MAX + 1.);
  }
  return static_cast<T>(std::numeric_limits<T>::max() * R);
}

template <typename T, typename std::enable_if<rank<T>() == 1>::type * = nullptr>
static inline void initVal(T &Val) {
  Val.x = getRandom<decltype(T::x)>();
}

template <typename T, typename std::enable_if<rank<T>() == 2>::type * = nullptr>
static inline void initVal(T &Val) {
  Val.x = getRandom<decltype(T::x)>();
  Val.y = getRandom<decltype(T::x)>();
}

template <typename T, typename std::enable_if<rank<T>() == 4>::type * = nullptr>
static inline void initVal(T &Val) {
  Val.x = getRandom<decltype(T::x)>();
  Val.y = getRandom<decltype(T::x)>();
  Val.z = getRandom<decltype(T::x)>();
  Val.w = getRandom<decltype(T::x)>();
}

template <typename T, typename std::enable_if<rank<T>() == 1>::type * = nullptr>
static inline void printVector(T &Val) {
  using B = decltype(T::x);
  constexpr bool IsChar =
      std::is_same<B, char>::value || std::is_same<B, unsigned char>::value;
  std::cout << "(";
  std::cout << (IsChar ? static_cast<int>(Val.x) : Val.x);
  std::cout << ")";
}

template <typename T, typename std::enable_if<rank<T>() == 2>::type * = nullptr>
static inline void printVector(T &Val) {
  using B = decltype(T::x);
  constexpr bool IsChar =
      std::is_same<B, char>::value || std::is_same<B, unsigned char>::value;
  std::cout << "(";
  std::cout << (IsChar ? static_cast<int>(Val.x) : Val.x);
  std::cout << ", " << (IsChar ? static_cast<int>(Val.y) : Val.y);
  std::cout << ")";
}

template <typename T, typename std::enable_if<rank<T>() == 4>::type * = nullptr>
static inline void printVector(T &Val) {
  using B = decltype(T::x);
  constexpr bool IsChar =
      std::is_same<B, char>::value || std::is_same<B, unsigned char>::value;
  std::cout << "(";
  std::cout << (IsChar ? static_cast<int>(Val.x) : Val.x);
  std::cout << ", " << (IsChar ? static_cast<int>(Val.y) : Val.y);
  std::cout << ", " << (IsChar ? static_cast<int>(Val.z) : Val.z);
  std::cout << ", " << (IsChar ? static_cast<int>(Val.w) : Val.w);
  std::cout << ")";
}

template <typename T, typename std::enable_if<rank<T>() == 1>::type * = nullptr>
static inline bool isEqual(const T &Val0, const T &Val1) {
  return Val0.x == Val1.x;
}

template <typename T, typename std::enable_if<rank<T>() == 2>::type * = nullptr>
static inline bool isEqual(const T &Val0, const T &Val1) {
  return Val0.x == Val1.x && Val0.y == Val1.y;
}

template <typename T, typename std::enable_if<rank<T>() == 4>::type * = nullptr>
static inline bool isEqual(const T &Val0, const T &Val1) {
  return Val0.x == Val1.x && Val0.y == Val1.y && Val0.z == Val1.z &&
         Val0.w == Val1.w;
}

template <typename T> bool runTest(const char *Description) {
  const int N = 1024;
  bool TestResult = true;
  // Allocating the required buffer on gpu device
  T *TexBuf, *TexBufOut;
  T Val[N], Output[N];
  printf("%s<%s>(): size: %zu, %zu\n", __FUNCTION__, Description, sizeof(T),
         sizeof(decltype(T::x)));

  memset(Output, 0, sizeof(Output));
  std::srand(
      std::time(nullptr)); // use current time as seed for random generator

  for (int i = 0; i < N; i++) {
    initVal<T>(Val[i]);
  }

  HIP_CHECK(hipMalloc(&TexBuf, N * sizeof(T)));
  HIP_CHECK(hipMalloc(&TexBufOut, N * sizeof(T)));
  HIP_CHECK(hipMemcpy(TexBuf, Val, N * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(TexBufOut, 0, N * sizeof(T)));
  hipResourceDesc ResDescLinear;

  memset(&ResDescLinear, 0, sizeof(ResDescLinear));
  ResDescLinear.resType = hipResourceTypeLinear;
  ResDescLinear.res.linear.devPtr = TexBuf;
  ResDescLinear.res.linear.desc = hipCreateChannelDesc<T>();
  ResDescLinear.res.linear.sizeInBytes = N * sizeof(T);

  hipTextureDesc TexDesc;
  memset(&TexDesc, 0, sizeof(TexDesc));
  TexDesc.readMode = hipReadModeElementType;
  TexDesc.addressMode[0] = hipAddressModeClamp;

  // Creating texture object
  hipTextureObject_t TexObj = 0;
  HIP_CHECK(hipCreateTextureObject(&TexObj, &ResDescLinear, &TexDesc, NULL));

  dim3 DimBlock(64, 1, 1);
  dim3 DimGrid((N + DimBlock.x - 1) / DimBlock.x, 1, 1);

  hipLaunchKernelGGL(tex1dKernelFetch<T>, DimGrid, DimBlock, 0, 0, TexBufOut,
                     TexObj, N);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(Output, TexBufOut, N * sizeof(T), hipMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    if (!isEqual(Output[i], Val[i])) {
      std::cout << "output[" << i << "]= ";
      printVector<T>(Output[i]);
      std::cout << ", expected val[" << i << "]= ";
      printVector<T>(Val[i]);
      std::cout << "\n";
      TestResult = false;
      break;
    }
  }
  HIP_CHECK(hipDestroyTextureObject(TexObj));
  HIP_CHECK(hipFree(TexBuf));
  HIP_CHECK(hipFree(TexBufOut));

  printf(": %s\n", TestResult ? "succeeded" : "failed");
  return TestResult;
}

#define CHK(X)                                                                 \
  if (!X)                                                                      \
  return 1

int main() {
  // test for char
  CHK(runTest<char1>("char1"));
  CHK(runTest<char2>("char2"));
  CHK(runTest<char4>("char4"));

  // test for uchar
  CHK(runTest<uchar1>("uchar1"));
  CHK(runTest<uchar2>("uchar2"));
  CHK(runTest<uchar4>("uchar4"));

  // test for short
  CHK(runTest<short1>("short1"));
  CHK(runTest<short2>("short2"));
  CHK(runTest<short4>("short4"));

  // test for ushort
  CHK(runTest<ushort1>("ushort1"));
  CHK(runTest<ushort2>("ushort2"));
  CHK(runTest<ushort4>("ushort4"));

  // test for int
  CHK(runTest<int1>("int1"));
  CHK(runTest<int2>("int2"));
  CHK(runTest<int4>("int4"));

  // test for unsigned int
  CHK(runTest<uint1>("uint1"));
  CHK(runTest<uint2>("uint2"));
  CHK(runTest<uint4>("uint4"));

  // test for float
  CHK(runTest<float1>("float1"));
  CHK(runTest<float2>("float2"));
  CHK(runTest<float4>("float4"));

  printf("PASSED\n");
  return 0;
}
