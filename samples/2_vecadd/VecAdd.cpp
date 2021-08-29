#include <iostream>
#include <iomanip>
#include <random>
#include <functional>
#include <cassert>
#include <chrono>

// hip header file
#include "hip/hip_runtime.h"

#define SEED 192849223

#define LOC_WG 16
#define GRID_WG 128
#define NUM (GRID_WG*LOC_WG)

/*****************************************************************************/

typedef std::function<float(void)> RandomGenFuncFloat;
typedef std::function<int(void)> RandomGenFuncInt;

template <typename T>
__global__ void
VecADD (const T * __restrict A, const T * __restrict B, T * __restrict C, const T multiplier)
{
  const uint i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  C[i] = A[i] + B[i] + multiplier;
}


/*****************************************************************************/

template <typename T>
__host__
void ArrayMADcpuReference(const T* __restrict A,
                          const T* __restrict B,
                          T * __restrict C,
                          const T multiplier) {
  for (uint i = 0; i < NUM; i++) {
    C[i] = A[i] + B[i] + multiplier;
  }
}

template <typename T, typename CMPT>
bool compareRes4(size_t i, bool print, T res1, T res2)
{
  if (res1.x == res2.x && res1.y == res2.y &&
      res1.z == res2.z && res1.w == res2.w)
    return true;

  if (print) {
      std::cerr << "FAIL AT: " << i << "\n";

      std::cerr << "CPU: " << res1.x << " "
                << res1.y << " "
                << res1.z << " "
                << res1.w << "\n";
      std::cerr << "GPU: " << res2.x << " "
                << res2.y << " "
                << res2.z << " "
                << res2.w << "\n";
    }
  return false;
}

template <typename T, typename CMPT>
bool compareRes3(size_t i, bool print, T res1, T res2) {
  if (res1.x == res2.x && res1.y == res2.y &&
      res1.z == res2.z)
    return true;

  if (print) {
    std::cerr << "FAIL AT: " << i << "\n";

    std::cerr << "CPU: " << res1.x << " " << res1.y << " " << res1.z << "\n";

    std::cerr << "GPU: " << res2.x << " " << res2.y << " " << res2.z << "\n";
  }
  return false;
}

template <typename T, typename CMPT>
bool compareRes2(size_t i, bool print, T res1, T res2)
{
  if (res1.x == res2.x && res1.y == res2.y)
    return true;

  if (print) {
      std::cerr << "FAIL AT: " << i << "\n";

      std::cerr << "CPU: " << res1.x << " "
                << res1.y << "\n";
      std::cerr << "GPU: " << res2.x << " "
                << res2.y << "\n";
  }
  return false;
}

/*
template <typename T, typename CMPT>
bool compareRes1(size_t i, bool print, T res1, T res2)
{
  CMPT res = (res1 != res2);

  if (res.x == 0)
    return true;

  if (print) {
      std::cerr << "FAIL AT: " << i << "\n";
      std::cerr << "CMP: " << res.x << "\n";

      std::cerr << "CPU: " << res1.x << "\n";
      std::cerr << "GPU: " << res2.x << "\n";
  }
  return false;
}
*/

#define ERR_CHECK_2 \
  do { \
  err = hipGetLastError(); \
    if (err != hipSuccess) { \
      std::cerr << "HIP API error\n"; \
      return -1; \
    } \
  } while (0)


#define ERR_CHECK \
  do { \
    if (err != hipSuccess) { \
      std::cerr << "HIP API error\n"; \
      return -1; \
    } \
  } while (0)


template <typename T, typename RNG>
__host__ int TestVectors(RNG rnd, const T multiplier,
                         bool (*comparator)(size_t i, bool print, T res1, T res2)
                         ) {
    hipError_t err;
    T* Array1;
    T* Array2;
    T* ResArray;
    T* cpuResArray;

    T* gpuArray1;
    T* gpuArray2;
    T* gpuResArray;

    Array1 = new T [NUM];
    Array2 = new T [NUM];
    ResArray = new T [NUM];
    cpuResArray = new T [NUM];

    // initialize the input data
    for (size_t i = 0; i < NUM; i++) {
        Array1[i] = (T)rnd();
        Array2[i] = (T)rnd();
    }

    err = hipMalloc((void**)&gpuArray1, NUM * sizeof(T));
    ERR_CHECK;
    err = hipMalloc((void**)&gpuArray2, NUM * sizeof(T));
    ERR_CHECK;
    err = hipMalloc((void**)&gpuResArray, NUM * sizeof(T));
    ERR_CHECK;

    err = hipMemcpy(gpuArray1, Array1, NUM * sizeof(T), hipMemcpyHostToDevice);
    ERR_CHECK;
    err = hipMemcpy(gpuArray2, Array2, NUM * sizeof(T), hipMemcpyHostToDevice);
    ERR_CHECK;
    hipLaunchKernelGGL(VecADD<T>,
                       dim3(GRID_WG),
                       dim3(LOC_WG),
                       0, 0,
                       gpuArray1, gpuArray2, gpuResArray, multiplier);
    ERR_CHECK_2;
    err = hipMemcpy(ResArray, gpuResArray, NUM * sizeof(T), hipMemcpyDeviceToHost);
    ERR_CHECK;

    ArrayMADcpuReference<T>(Array1, Array2, cpuResArray, multiplier);

    size_t failures = 0;
    for (size_t i = 0; i < NUM; i++) {
        if (comparator(i, (failures < 50), cpuResArray[i], ResArray[i]))
          continue;
        ++failures;
      }

    if (failures > 0) {
        std::cout << "FAIL: " << failures << " failures \n";
      }
    else {
        std::cout << "PASSED\n";
      }

    // free the resources on device side
    err = hipFree(gpuArray1);
    ERR_CHECK;
    err = hipFree(gpuArray2);
    ERR_CHECK;
    err = hipFree(gpuResArray);
    ERR_CHECK;

    // free the resources on host side
    delete [] Array1;
    delete [] Array2;
    delete [] ResArray;
    delete [] cpuResArray;

    return 0;
}


int main() {

  hipError_t err;

  std::mt19937 gen(SEED);
  RandomGenFuncFloat rndf = std::bind(std::uniform_real_distribution<float>{100.0f, 1000.0f}, gen);
  RandomGenFuncInt rndi = std::bind(std::uniform_int_distribution<int>{100, 1000}, gen);
  //RandomGenFunc a = rnd;
  //std::function<float(void)> fun = rnd;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  std::cerr << std::hexfloat;

  std::cout << "Device name " << devProp.name << std::endl;

  std::cout << "float4 test\n";
  float4 m_f4 = make_float4(7.0f, 7.0f, 7.0f, 7.0f);
  TestVectors<float4, RandomGenFuncFloat>(rndf, m_f4, compareRes2<float4, int4>);

  std::cout << "float3 test\n";
  float3 m_f3 = make_float3(7.0f, 7.0f, 7.0f);
  TestVectors<float3, RandomGenFuncFloat>(rndf, m_f3,
                                          compareRes2<float3, int3>);

  std::cout << "float2 test\n";
  float2 m_f2 = make_float2(7.0f, 7.0f);
  TestVectors<float2, RandomGenFuncFloat>(rndf, m_f2,
                                          compareRes2<float2, int2>);

  // std::cout << "float1 test\n";
  // float1 m_f1 = make_float1(22);
  // TestVectors<float1, RandomGenFuncInt>(rndi, m_f1, compareRes1<float1,
  // int1>);

  std::cout << "int4 test\n";
  int4 m_i4 = make_int4(3, 17, 48, 29);
  TestVectors<int4, RandomGenFuncInt>(rndi, m_i4, compareRes4<int4, int4>);

  std::cout << "int3 test\n";
  int3 m_i3 = make_int3(3, 17, 38);
  TestVectors<int3, RandomGenFuncInt>(rndi, m_i3, compareRes3<int3, int3>);

  std::cout << "int2 test\n";
  int2 m_i2 = make_int2(22, 19);
  TestVectors<int2, RandomGenFuncInt>(rndi, m_i2, compareRes2<int2, int2>);

  // std::cout << "uint1 test\n";
  // uint1 m_i1 = make_uint1(22);
  // TestVectors<uint1, RandomGenFuncInt>(rndi, m_i1, compareRes1<uint1, int1>);

  std::cout << "ulong4 test\n";
  ulong4 m_ul4 = make_ulong4(3, 17, 48, 29);
  TestVectors<ulong4, RandomGenFuncInt>(rndi, m_ul4,
                                        compareRes4<ulong4, long4>);

  std::cout << "ulong3 test\n";
  ulong3 m_ul3 = make_ulong3(3, 17, 38);
  TestVectors<ulong3, RandomGenFuncInt>(rndi, m_ul3,
                                        compareRes3<ulong3, long3>);

  std::cout << "ulong2 test\n";
  ulong2 m_ul2 = make_ulong2(22, 19);
  TestVectors<ulong2, RandomGenFuncInt>(rndi, m_ul2,
                                        compareRes2<ulong2, long2>);

  // std::cout << "ulong1 test\n";
  // ulong1 m_ul1 = make_ulong1(22);
  // TestVectors<ulong1, RandomGenFuncInt>(rndi, m_ul1, compareRes1<ulong1,
  // long1>);

  std::cout << "ulonglong4 test\n";
  ulonglong4 m_ull4 = make_ulonglong4(3, 17, 48, 29);
  TestVectors<ulonglong4, RandomGenFuncInt>(rndi, m_ull4,
                                            compareRes4<ulonglong4, long4>);

  std::cout << "ulonglong3 test\n";
  ulonglong3 m_ull3 = make_ulonglong3(3, 17, 38);
  TestVectors<ulonglong3, RandomGenFuncInt>(rndi, m_ull3,
                                            compareRes3<ulonglong3, long3>);

  std::cout << "ulonglong2 test\n";
  ulonglong2 m_ull2 = make_ulonglong2(22, 19);
  TestVectors<ulonglong2, RandomGenFuncInt>(rndi, m_ull2,
                                            compareRes2<ulonglong2, long2>);

  // std::cout << "ulonglong1 test\n";
  // ulonglong1 m_ull1 = make_ulonglong1(22);
  // TestVectors<ulonglong1, RandomGenFuncInt>(rndi, m_ull1,
  // compareRes1<ulonglong1, long1>);

  std::cout << "uchar4 test\n";
  uchar4 m_uc4 = make_uchar4(3, 17, 48, 29);
  TestVectors<uchar4, RandomGenFuncInt>(rndi, m_uc4,
                                        compareRes4<uchar4, char4>);

  std::cout << "uchar3 test\n";
  uchar3 m_uc3 = make_uchar3(3, 17, 38);
  TestVectors<uchar3, RandomGenFuncInt>(rndi, m_uc3,
                                        compareRes3<uchar3, char3>);

  std::cout << "uchar2 test\n";
  uchar2 m_uc2 = make_uchar2(22, 19);
  TestVectors<uchar2, RandomGenFuncInt>(rndi, m_uc2,
                                        compareRes2<uchar2, char2>);

  // std::cout << "uchar1 test\n";
  // uchar1 m_uc1 = make_uchar1(11);
  // TestVectors<uchar1, RandomGenFuncInt>(rndi, m_uc1, compareRes1<uchar1,
  // char1>);
}
