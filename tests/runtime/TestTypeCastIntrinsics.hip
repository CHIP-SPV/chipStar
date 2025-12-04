// Tests for type casting intrinsics with rounding modes
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <iostream>

#define CHECK_HIP(expr)                                                        \
  do {                                                                         \
    hipError_t err = (expr);                                                   \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Kernel to test double2float conversions
__global__ void testDouble2Float(double input, float *rd, float *rn, float *ru,
                                 float *rz) {
  *rd = __double2float_rd(input);
  *rn = __double2float_rn(input);
  *ru = __double2float_ru(input);
  *rz = __double2float_rz(input);
}

// Kernel to test int2float conversions
__global__ void testInt2Float(int input, float *rd, float *rn, float *ru,
                              float *rz) {
  *rd = __int2float_rd(input);
  *rn = __int2float_rn(input);
  *ru = __int2float_ru(input);
  *rz = __int2float_rz(input);
}

// Kernel to test uint2float conversions
__global__ void testUint2Float(unsigned int input, float *rd, float *rn,
                               float *ru, float *rz) {
  *rd = __uint2float_rd(input);
  *rn = __uint2float_rn(input);
  *ru = __uint2float_ru(input);
  *rz = __uint2float_rz(input);
}

// Kernel to test ll2float conversions
__global__ void testLL2Float(long long int input, float *rd, float *rn,
                             float *ru, float *rz) {
  *rd = __ll2float_rd(input);
  *rn = __ll2float_rn(input);
  *ru = __ll2float_ru(input);
  *rz = __ll2float_rz(input);
}

// Kernel to test ull2float conversions
__global__ void testULL2Float(unsigned long long int input, float *rd,
                              float *rn, float *ru, float *rz) {
  *rd = __ull2float_rd(input);
  *rn = __ull2float_rn(input);
  *ru = __ull2float_ru(input);
  *rz = __ull2float_rz(input);
}

// Kernel to test float2int conversions
__global__ void testFloat2Int(float input, int *rd, int *rn, int *ru, int *rz) {
  *rd = __float2int_rd(input);
  *rn = __float2int_rn(input);
  *ru = __float2int_ru(input);
  *rz = __float2int_rz(input);
}

// Kernel to test double2int conversions
__global__ void testDouble2Int(double input, int *rd, int *rn, int *ru,
                               int *rz) {
  *rd = __double2int_rd(input);
  *rn = __double2int_rn(input);
  *ru = __double2int_ru(input);
  *rz = __double2int_rz(input);
}

// Kernel to test hiloint2double and double2loint/double2hiint
__global__ void testHiLoInt(int hi, int lo, double *result, int *lo_out,
                            int *hi_out) {
  *result = __hiloint2double(hi, lo);
  *lo_out = __double2loint(*result);
  *hi_out = __double2hiint(*result);
}

// Kernel to test ll2double conversions
__global__ void testLL2Double(long long int input, double *rd, double *rn,
                              double *ru, double *rz) {
  *rd = __ll2double_rd(input);
  *rn = __ll2double_rn(input);
  *ru = __ll2double_ru(input);
  *rz = __ll2double_rz(input);
}

// Kernel to test ull2double conversions
__global__ void testULL2Double(unsigned long long int input, double *rd,
                               double *rn, double *ru, double *rz) {
  *rd = __ull2double_rd(input);
  *rn = __ull2double_rn(input);
  *ru = __ull2double_ru(input);
  *rz = __ull2double_rz(input);
}

// Kernel to test int2double_rn and uint2double_rn
__global__ void testInt2Double(int input_int, unsigned int input_uint,
                               double *int_result, double *uint_result) {
  *int_result = __int2double_rn(input_int);
  *uint_result = __uint2double_rn(input_uint);
}

// Kernel to test float2ll conversions
__global__ void testFloat2LL(float input, long long int *rd, long long int *rn,
                             long long int *ru, long long int *rz) {
  *rd = __float2ll_rd(input);
  *rn = __float2ll_rn(input);
  *ru = __float2ll_ru(input);
  *rz = __float2ll_rz(input);
}

// Kernel to test double2ll conversions
__global__ void testDouble2LL(double input, long long int *rd,
                              long long int *rn, long long int *ru,
                              long long int *rz) {
  *rd = __double2ll_rd(input);
  *rn = __double2ll_rn(input);
  *ru = __double2ll_ru(input);
  *rz = __double2ll_rz(input);
}

// Kernel to test float2uint conversions
__global__ void testFloat2Uint(float input, unsigned int *rd, unsigned int *rn,
                               unsigned int *ru, unsigned int *rz) {
  *rd = __float2uint_rd(input);
  *rn = __float2uint_rn(input);
  *ru = __float2uint_ru(input);
  *rz = __float2uint_rz(input);
}

// Kernel to test double2uint conversions
__global__ void testDouble2Uint(double input, unsigned int *rd,
                                unsigned int *rn, unsigned int *ru,
                                unsigned int *rz) {
  *rd = __double2uint_rd(input);
  *rn = __double2uint_rn(input);
  *ru = __double2uint_ru(input);
  *rz = __double2uint_rz(input);
}

// Kernel to test float2ull conversions
__global__ void testFloat2ULL(float input, unsigned long long int *rd,
                              unsigned long long int *rn,
                              unsigned long long int *ru,
                              unsigned long long int *rz) {
  *rd = __float2ull_rd(input);
  *rn = __float2ull_rn(input);
  *ru = __float2ull_ru(input);
  *rz = __float2ull_rz(input);
}

// Kernel to test double2ull conversions
__global__ void testDouble2ULL(double input, unsigned long long int *rd,
                               unsigned long long int *rn,
                               unsigned long long int *ru,
                               unsigned long long int *rz) {
  *rd = __double2ull_rd(input);
  *rn = __double2ull_rn(input);
  *ru = __double2ull_ru(input);
  *rz = __double2ull_rz(input);
}

int testDouble2FloatConversions() {
  float *d_rd, *d_rn, *d_ru, *d_rz;
  float h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(float)));

  // Test with a value that has more precision than float can represent
  // 1.0000001192092896 is between two representable floats
  double testVal = 1.0000001192092896;

  testDouble2Float<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rn, d_rn, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(float), hipMemcpyDeviceToHost));

  // rd should round towards negative (smaller value)
  // ru should round towards positive (larger value)
  // rz should round towards zero
  // For positive values: rd <= rz <= rn (or ru)
  if (!(h_rd <= h_rn && h_rn <= h_ru)) {
    std::cerr << "FAILED: double2float rounding order incorrect" << std::endl;
    std::cerr << "  rd=" << h_rd << " rn=" << h_rn << " ru=" << h_ru
              << std::endl;
    return 1;
  }

  // Test with negative value
  testVal = -1.0000001192092896;
  testDouble2Float<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rn, d_rn, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(float), hipMemcpyDeviceToHost));

  // For negative values: rd <= rn <= ru (rd is most negative)
  if (!(h_rd <= h_rn && h_rn <= h_ru)) {
    std::cerr << "FAILED: double2float negative rounding order incorrect"
              << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  double2float: OK" << std::endl;
  return 0;
}

int testInt2FloatConversions() {
  float *d_rd, *d_rn, *d_ru, *d_rz;
  float h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(float)));

  // 16777217 = 2^24 + 1, cant be exactly represented in float
  // float can only represent 2^24 or 2^24 + 2
  int testVal = 16777217;

  testInt2Float<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rn, d_rn, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(float), hipMemcpyDeviceToHost));

  float expected_rd = 16777216.0f; // rounds down to 2^24
  float expected_ru = 16777218.0f; // rounds up to 2^24 + 2
  float expected_rz = 16777216.0f; // towards zero = down for positive

  if (h_rd != expected_rd) {
    std::cerr << "FAILED: int2float_rd expected " << expected_rd << " got "
              << h_rd << std::endl;
    return 1;
  }
  if (h_ru != expected_ru) {
    std::cerr << "FAILED: int2float_ru expected " << expected_ru << " got "
              << h_ru << std::endl;
    return 1;
  }
  if (h_rz != expected_rz) {
    std::cerr << "FAILED: int2float_rz expected " << expected_rz << " got "
              << h_rz << std::endl;
    return 1;
  }

  // Test negative value
  testVal = -16777217;
  testInt2Float<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(float), hipMemcpyDeviceToHost));

  expected_rd = -16777218.0f; // rounds down (more negative)
  expected_rz = -16777216.0f; // towards zero = up for negative

  if (h_rd != expected_rd) {
    std::cerr << "FAILED: int2float_rd negative expected " << expected_rd
              << " got " << h_rd << std::endl;
    return 1;
  }
  if (h_rz != expected_rz) {
    std::cerr << "FAILED: int2float_rz negative expected " << expected_rz
              << " got " << h_rz << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  int2float: OK" << std::endl;
  return 0;
}

int testUint2FloatConversions() {
  float *d_rd, *d_rn, *d_ru, *d_rz;
  float h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(float)));

  unsigned int testVal = 16777217u;

  testUint2Float<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(float), hipMemcpyDeviceToHost));

  float expected_rd = 16777216.0f;
  float expected_ru = 16777218.0f;
  float expected_rz = 16777216.0f;

  if (h_rd != expected_rd || h_ru != expected_ru || h_rz != expected_rz) {
    std::cerr << "FAILED: uint2float conversions incorrect" << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  uint2float: OK" << std::endl;
  return 0;
}

int testLL2FloatConversions() {
  float *d_rd, *d_rn, *d_ru, *d_rz;
  float h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(float)));

  long long int testVal = 16777217LL;

  testLL2Float<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(float), hipMemcpyDeviceToHost));

  float expected_rd = 16777216.0f;
  float expected_ru = 16777218.0f;
  float expected_rz = 16777216.0f;

  if (h_rd != expected_rd || h_ru != expected_ru || h_rz != expected_rz) {
    std::cerr << "FAILED: ll2float conversions incorrect" << std::endl;
    std::cerr << "  rd=" << h_rd << " (expected " << expected_rd << ")"
              << std::endl;
    std::cerr << "  ru=" << h_ru << " (expected " << expected_ru << ")"
              << std::endl;
    std::cerr << "  rz=" << h_rz << " (expected " << expected_rz << ")"
              << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  ll2float: OK" << std::endl;
  return 0;
}

int testULL2FloatConversions() {
  float *d_rd, *d_rn, *d_ru, *d_rz;
  float h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(float)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(float)));

  unsigned long long int testVal = 16777217ULL;

  testULL2Float<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(float), hipMemcpyDeviceToHost));

  float expected_rd = 16777216.0f;
  float expected_ru = 16777218.0f;
  float expected_rz = 16777216.0f;

  if (h_rd != expected_rd || h_ru != expected_ru || h_rz != expected_rz) {
    std::cerr << "FAILED: ull2float conversions incorrect" << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  ull2float: OK" << std::endl;
  return 0;
}

int testFloat2IntConversions() {
  int *d_rd, *d_rn, *d_ru, *d_rz;
  int h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(int)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(int)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(int)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(int)));

  float testVal = 2.5f;

  testFloat2Int<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rn, d_rn, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(int), hipMemcpyDeviceToHost));

  if (h_rd != 2 || h_rn != 2 || h_ru != 3 || h_rz != 2) {
    std::cerr << "FAILED: float2int(2.5f) incorrect" << std::endl;
    std::cerr << "  rd=" << h_rd << " rn=" << h_rn << " ru=" << h_ru
              << " rz=" << h_rz << std::endl;
    return 1;
  }

  // Test negative
  testVal = -2.5f;
  testFloat2Int<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rn, d_rn, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(int), hipMemcpyDeviceToHost));

  if (h_rd != -3 || h_rn != -2 || h_ru != -2 || h_rz != -2) {
    std::cerr << "FAILED: float2int(-2.5f) incorrect" << std::endl;
    std::cerr << "  rd=" << h_rd << " rn=" << h_rn << " ru=" << h_ru
              << " rz=" << h_rz << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  float2int: OK" << std::endl;
  return 0;
}

int testDouble2IntConversions() {
  int *d_rd, *d_rn, *d_ru, *d_rz;
  int h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(int)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(int)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(int)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(int)));

  double testVal = 2.5;

  testDouble2Int<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rn, d_rn, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(int), hipMemcpyDeviceToHost));

  if (h_rd != 2 || h_rn != 2 || h_ru != 3 || h_rz != 2) {
    std::cerr << "FAILED: double2int(2.5) incorrect" << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  double2int: OK" << std::endl;
  return 0;
}

int testHiLoIntConversions() {
  double *d_result;
  int *d_lo, *d_hi;
  double h_result;
  int h_lo, h_hi;

  CHECK_HIP(hipMalloc(&d_result, sizeof(double)));
  CHECK_HIP(hipMalloc(&d_lo, sizeof(int)));
  CHECK_HIP(hipMalloc(&d_hi, sizeof(int)));

  // Use a known double value and extract/reconstruct
  int test_hi = 0x40000000; // exponent for 2.0
  int test_lo = 0x00000000;

  testHiLoInt<<<1, 1>>>(test_hi, test_lo, d_result, d_lo, d_hi);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_result, d_result, sizeof(double), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_lo, d_lo, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_hi, d_hi, sizeof(int), hipMemcpyDeviceToHost));

  if (h_result != 2.0 || h_lo != test_lo || h_hi != test_hi) {
    std::cerr << "FAILED: hiloint2double/double2loint/double2hiint incorrect"
              << std::endl;
    std::cerr << "  result=" << h_result << " (expected 2.0)" << std::endl;
    std::cerr << "  hi=" << std::hex << h_hi << " lo=" << h_lo << std::dec
              << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_result));
  CHECK_HIP(hipFree(d_lo));
  CHECK_HIP(hipFree(d_hi));

  std::cout << "  hiloint2double/double2loint/double2hiint: OK" << std::endl;
  return 0;
}

int testLL2DoubleConversions() {
  double *d_rd, *d_rn, *d_ru, *d_rz;
  double h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(double)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(double)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(double)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(double)));

  // 2^53 + 1 cant be exactly represented in double
  long long int testVal = (1LL << 53) + 1;

  testLL2Double<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(double), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rn, d_rn, sizeof(double), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(double), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(double), hipMemcpyDeviceToHost));

  // rd should be <= rn <= ru
  if (!(h_rd <= h_rn && h_rn <= h_ru)) {
    std::cerr << "FAILED: ll2double rounding order incorrect" << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  ll2double: OK" << std::endl;
  return 0;
}

int testULL2DoubleConversions() {
  double *d_rd, *d_rn, *d_ru, *d_rz;
  double h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(double)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(double)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(double)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(double)));

  unsigned long long int testVal = (1ULL << 53) + 1;

  testULL2Double<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(double), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rn, d_rn, sizeof(double), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(double), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(double), hipMemcpyDeviceToHost));

  if (!(h_rd <= h_rn && h_rn <= h_ru)) {
    std::cerr << "FAILED: ull2double rounding order incorrect" << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  ull2double: OK" << std::endl;
  return 0;
}

int testInt2DoubleConversions() {
  double *d_int_result, *d_uint_result;
  double h_int_result, h_uint_result;

  CHECK_HIP(hipMalloc(&d_int_result, sizeof(double)));
  CHECK_HIP(hipMalloc(&d_uint_result, sizeof(double)));

  int test_int = -12345;
  unsigned int test_uint = 12345u;

  testInt2Double<<<1, 1>>>(test_int, test_uint, d_int_result, d_uint_result);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(
      hipMemcpy(&h_int_result, d_int_result, sizeof(double), hipMemcpyDeviceToHost));
  CHECK_HIP(
      hipMemcpy(&h_uint_result, d_uint_result, sizeof(double), hipMemcpyDeviceToHost));

  if (h_int_result != -12345.0 || h_uint_result != 12345.0) {
    std::cerr << "FAILED: int2double_rn/uint2double_rn incorrect" << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_int_result));
  CHECK_HIP(hipFree(d_uint_result));

  std::cout << "  int2double_rn/uint2double_rn: OK" << std::endl;
  return 0;
}

int testFloat2LLConversions() {
  long long int *d_rd, *d_rn, *d_ru, *d_rz;
  long long int h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(long long int)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(long long int)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(long long int)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(long long int)));

  float testVal = 2.5f;

  testFloat2LL<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(long long int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rn, d_rn, sizeof(long long int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(long long int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(long long int), hipMemcpyDeviceToHost));

  if (h_rd != 2 || h_rn != 2 || h_ru != 3 || h_rz != 2) {
    std::cerr << "FAILED: float2ll(2.5f) incorrect" << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  float2ll: OK" << std::endl;
  return 0;
}

int testDouble2LLConversions() {
  long long int *d_rd, *d_rn, *d_ru, *d_rz;
  long long int h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(long long int)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(long long int)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(long long int)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(long long int)));

  double testVal = 2.5;

  testDouble2LL<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(long long int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rn, d_rn, sizeof(long long int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(long long int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(long long int), hipMemcpyDeviceToHost));

  if (h_rd != 2 || h_rn != 2 || h_ru != 3 || h_rz != 2) {
    std::cerr << "FAILED: double2ll(2.5) incorrect" << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  double2ll: OK" << std::endl;
  return 0;
}

int testFloat2UintConversions() {
  unsigned int *d_rd, *d_rn, *d_ru, *d_rz;
  unsigned int h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(unsigned int)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(unsigned int)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(unsigned int)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(unsigned int)));

  float testVal = 2.5f;

  testFloat2Uint<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(unsigned int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rn, d_rn, sizeof(unsigned int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(unsigned int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(unsigned int), hipMemcpyDeviceToHost));

  if (h_rd != 2 || h_rn != 2 || h_ru != 3 || h_rz != 2) {
    std::cerr << "FAILED: float2uint(2.5f) incorrect" << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  float2uint: OK" << std::endl;
  return 0;
}

int testDouble2UintConversions() {
  unsigned int *d_rd, *d_rn, *d_ru, *d_rz;
  unsigned int h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(unsigned int)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(unsigned int)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(unsigned int)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(unsigned int)));

  double testVal = 2.5;

  testDouble2Uint<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(&h_rd, d_rd, sizeof(unsigned int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rn, d_rn, sizeof(unsigned int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_ru, d_ru, sizeof(unsigned int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_rz, d_rz, sizeof(unsigned int), hipMemcpyDeviceToHost));

  if (h_rd != 2 || h_rn != 2 || h_ru != 3 || h_rz != 2) {
    std::cerr << "FAILED: double2uint(2.5) incorrect" << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  double2uint: OK" << std::endl;
  return 0;
}

int testFloat2ULLConversions() {
  unsigned long long int *d_rd, *d_rn, *d_ru, *d_rz;
  unsigned long long int h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(unsigned long long int)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(unsigned long long int)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(unsigned long long int)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(unsigned long long int)));

  float testVal = 2.5f;

  testFloat2ULL<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(
      hipMemcpy(&h_rd, d_rd, sizeof(unsigned long long int), hipMemcpyDeviceToHost));
  CHECK_HIP(
      hipMemcpy(&h_rn, d_rn, sizeof(unsigned long long int), hipMemcpyDeviceToHost));
  CHECK_HIP(
      hipMemcpy(&h_ru, d_ru, sizeof(unsigned long long int), hipMemcpyDeviceToHost));
  CHECK_HIP(
      hipMemcpy(&h_rz, d_rz, sizeof(unsigned long long int), hipMemcpyDeviceToHost));

  if (h_rd != 2 || h_rn != 2 || h_ru != 3 || h_rz != 2) {
    std::cerr << "FAILED: float2ull(2.5f) incorrect" << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  float2ull: OK" << std::endl;
  return 0;
}

int testDouble2ULLConversions() {
  unsigned long long int *d_rd, *d_rn, *d_ru, *d_rz;
  unsigned long long int h_rd, h_rn, h_ru, h_rz;

  CHECK_HIP(hipMalloc(&d_rd, sizeof(unsigned long long int)));
  CHECK_HIP(hipMalloc(&d_rn, sizeof(unsigned long long int)));
  CHECK_HIP(hipMalloc(&d_ru, sizeof(unsigned long long int)));
  CHECK_HIP(hipMalloc(&d_rz, sizeof(unsigned long long int)));

  double testVal = 2.5;

  testDouble2ULL<<<1, 1>>>(testVal, d_rd, d_rn, d_ru, d_rz);
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(
      hipMemcpy(&h_rd, d_rd, sizeof(unsigned long long int), hipMemcpyDeviceToHost));
  CHECK_HIP(
      hipMemcpy(&h_rn, d_rn, sizeof(unsigned long long int), hipMemcpyDeviceToHost));
  CHECK_HIP(
      hipMemcpy(&h_ru, d_ru, sizeof(unsigned long long int), hipMemcpyDeviceToHost));
  CHECK_HIP(
      hipMemcpy(&h_rz, d_rz, sizeof(unsigned long long int), hipMemcpyDeviceToHost));

  if (h_rd != 2 || h_rn != 2 || h_ru != 3 || h_rz != 2) {
    std::cerr << "FAILED: double2ull(2.5) incorrect" << std::endl;
    return 1;
  }

  CHECK_HIP(hipFree(d_rd));
  CHECK_HIP(hipFree(d_rn));
  CHECK_HIP(hipFree(d_ru));
  CHECK_HIP(hipFree(d_rz));

  std::cout << "  double2ull: OK" << std::endl;
  return 0;
}

int main() {
  std::cout << "Testing type casting intrinsics..." << std::endl;

  if (testDouble2FloatConversions())
    return 1;
  if (testInt2FloatConversions())
    return 1;
  if (testUint2FloatConversions())
    return 1;
  if (testLL2FloatConversions())
    return 1;
  if (testULL2FloatConversions())
    return 1;
  if (testFloat2IntConversions())
    return 1;
  if (testDouble2IntConversions())
    return 1;
  if (testHiLoIntConversions())
    return 1;
  if (testLL2DoubleConversions())
    return 1;
  if (testULL2DoubleConversions())
    return 1;
  if (testInt2DoubleConversions())
    return 1;
  if (testFloat2LLConversions())
    return 1;
  if (testDouble2LLConversions())
    return 1;
  if (testFloat2UintConversions())
    return 1;
  if (testDouble2UintConversions())
    return 1;
  if (testFloat2ULLConversions())
    return 1;
  if (testDouble2ULLConversions())
    return 1;

  std::cout << "PASSED" << std::endl;
  return 0;
}

