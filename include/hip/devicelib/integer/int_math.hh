#ifndef HIP_INCLUDE_DEVICELIB_INT_MATH_H
#define HIP_INCLUDE_DEVICELIB_INT_MATH_H

#include <hip/devicelib/macros.hh>

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" {
NON_OVLD int GEN_NAME2(max, i)(int a, int b);
NON_OVLD unsigned GEN_NAME2(max, u)(unsigned int a, unsigned int b);
NON_OVLD long int GEN_NAME2(max, l)(long int a, long int b);
NON_OVLD unsigned long int GEN_NAME2(max, ul)(unsigned long int a,
                                              unsigned long int b);

NON_OVLD int GEN_NAME2(min, i)(int a, int b);
NON_OVLD unsigned GEN_NAME2(min, u)(unsigned int a, unsigned int b);
NON_OVLD long int GEN_NAME2(min, l)(long int a, long int b);
NON_OVLD unsigned long int GEN_NAME2(min, ul)(unsigned long int a,
                                              unsigned long int b);

NON_OVLD int GEN_NAME2(abs, i)(int a);
NON_OVLD long int GEN_NAME2(abs, l)(long int a);
}

EXPORT OVLD int abs(int a) { return GEN_NAME2(abs, i)(a); }
EXPORT OVLD long int labs(long int a) { return GEN_NAME2(abs, l)(a); }

EXPORT OVLD unsigned long int max(const unsigned long int a,
                                  const unsigned long int b) {
  return GEN_NAME2(max, ul)(a, b);
}
EXPORT OVLD unsigned long int max(const unsigned long int a, const long int b) {
  return (b < 0) ? a : GEN_NAME2(max, ul)(a, (unsigned long)b);
}
EXPORT OVLD unsigned long int max(const long int a, const unsigned long int b) {
  return max(b, a);
}
EXPORT OVLD long int max(const long int a, const long int b) {
  return GEN_NAME2(max, l)(a, b);
}

EXPORT OVLD unsigned int max(const unsigned int a, const unsigned int b) {
  return GEN_NAME2(max, u)(a, b);
}
EXPORT OVLD int max(const int a, const int b) {
  return GEN_NAME2(max, i)(a, b);
}
EXPORT OVLD unsigned int max(const unsigned int a, const int b) {
  return (b < 0) ? a : GEN_NAME2(max, u)(a, (unsigned)b);
}
EXPORT OVLD unsigned int max(const int a, const unsigned int b) {
  return max(b, a);
}

EXPORT OVLD unsigned long int min(const unsigned long int a,
                                  const unsigned long int b) {
  return GEN_NAME2(min, ul)(a, b);
}

EXPORT OVLD unsigned long int min(const unsigned long int a, const long int b) {
  return (b < 0) ? a : GEN_NAME2(min, ul)(a, (unsigned long)b);
}
EXPORT OVLD unsigned long int min(const long int a, const unsigned long int b) {
  return min(b, a);
}
EXPORT OVLD long int min(const long int a, const long int b) {
  return GEN_NAME2(min, l)(a, b);
}

EXPORT OVLD unsigned int min(const unsigned int a, const unsigned int b) {
  return GEN_NAME2(min, u)(a, b);
}
EXPORT OVLD int min(const int a, const int b) {
  return GEN_NAME2(min, i)(a, b);
}
EXPORT OVLD unsigned int min(const unsigned int a, const int b) {
  return (b < 0) ? a : GEN_NAME2(min, u)(a, (unsigned)b);
}
EXPORT OVLD unsigned int min(const int a, const unsigned int b) {
  return min(b, a);
}

EXPORT OVLD unsigned int umax(const unsigned int a, const unsigned int b) {
  return GEN_NAME2(max, u)(a, b);
}
EXPORT OVLD unsigned int umin(const unsigned int a, const unsigned int b) {
  return GEN_NAME2(min, u)(a, b);
}

#else
EXPORT OVLD int abs(int a);
EXPORT OVLD long int labs(long int a);
EXPORT OVLD unsigned long int max(const unsigned long int a, const long int b);
EXPORT OVLD unsigned long int max(const long int a, const unsigned long int b);
EXPORT OVLD unsigned long int max(const unsigned long int a,
                                  const unsigned long int b);
EXPORT OVLD long int max(const long int a, const long int b);
EXPORT OVLD unsigned int max(const unsigned int a, const int b);
EXPORT OVLD unsigned int max(const int a, const unsigned int b);
EXPORT OVLD unsigned int max(const unsigned int a, const unsigned int b);
EXPORT OVLD int max(const int a, const int b);
EXPORT OVLD unsigned long int min(const unsigned long int a, const long int b);
EXPORT OVLD unsigned long int min(const long int a, const unsigned long int b);
EXPORT OVLD unsigned long int min(const unsigned long int a,
                                  const unsigned long int b);
EXPORT OVLD long int min(const long int a, const long int b);
EXPORT OVLD unsigned int min(const unsigned int a, const int b);
EXPORT OVLD unsigned int min(const int a, const unsigned int b);
EXPORT OVLD unsigned int min(const unsigned int a, const unsigned int b);
EXPORT OVLD int min(const int a, const int b);

EXPORT OVLD unsigned int umax(const unsigned int a, const unsigned int b);
EXPORT OVLD unsigned int umin(const unsigned int a, const unsigned int b);
#endif


// __device__ long long int 	llabs ( long long int a )
// __device__ long long int 	llmax ( const long long int a, const long long
// int b )
// __device__ long long int 	llmin ( const long long int a, const long long
// int b )

// __device__ unsigned long long int 	ullmax ( const unsigned long long int a,
// const unsigned long long int b )
// __device__ unsigned long long int 	ullmin ( const unsigned long long int a,
// const unsigned long long int b )

#endif // include guard
