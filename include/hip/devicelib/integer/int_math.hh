#ifndef HIP_INCLUDE_DEVICELIB_INT_MATH_H
#define HIP_INCLUDE_DEVICELIB_INT_MATH_H

#if defined(__HIP_DEVICE_COMPILE__)

extern "C++" {

extern __device__ int max(int a, int b);
extern __device__ unsigned max(unsigned int a, unsigned int b);
extern __device__ long int max(long int a, long int b);
extern __device__ unsigned long int max(unsigned long int a,
                                        unsigned long int b);
extern __device__ int min(int a, int b);
extern __device__ unsigned min(unsigned int a, unsigned int b);
extern __device__ long int min(long int a, long int b);
extern __device__ unsigned long int min(unsigned long int a,
                                        unsigned long int b);

extern __device__ int abs(int a);
extern __device__ long int abs(long int a);

__device__ long int labs(long int a) { return abs(a); }
__device__ long long int abs(long long int a) { return abs((long int)a); }
__device__ long long int llabs(long long int a) { return abs((long int)a); }

__device__ unsigned long int max(const unsigned long int a, const long int b) {
  return (b < 0) ? a : max(a, (unsigned long)b);
}
__device__ unsigned long int max(const long int a, const unsigned long int b) {
  return max(b, a);
}

__device__ unsigned int max(const unsigned int a, const int b) {
  return (b < 0) ? a : max(a, (unsigned)b);
}
__device__ unsigned int max(const int a, const unsigned int b) {
  return max(b, a);
}

__device__ unsigned long int min(const unsigned long int a, const long int b) {
  return (b < 0) ? 0 : min(a, (unsigned long)b);
}
__device__ unsigned long int min(const long int a, const unsigned long int b) {
  return min(b, a);
}

__device__ unsigned int min(const unsigned int a, const int b) {
  return (b < 0) ? 0 : min(a, (unsigned)b);
}
__device__ unsigned int min(const int a, const unsigned int b) {
  return min(b, a);
}

__device__ unsigned int umax(const unsigned int a, const unsigned int b) {
  return max(a, b);
}

__device__ unsigned int umin(const unsigned int a, const unsigned int b) {
  return min(a, b);
}

__device__ unsigned long long int max(unsigned long long int a,
                                      unsigned long long int b) {
  return max((unsigned long)a, (unsigned long)b);
}

__device__ unsigned long long int min(unsigned long long int a,
                                      unsigned long long int b) {
  return min((unsigned long)a, (unsigned long)b);
}

__device__ unsigned long long int ullmax(const unsigned long long int a,
                                         const unsigned long long int b) {
  return max((unsigned long)a, (unsigned long)b);
}

__device__ unsigned long long int ullmin(const unsigned long long int a,
                                         const unsigned long long int b) {
  return min((unsigned long)a, (unsigned long)b);
}

__device__ long long int llmax(const long long int a, const long long int b) {
  return max((long)a, (long)b);
}

__device__ long long int llmin(const long long int a, const long long int b) {
  return min((long)a, (long)b);
}
}

#else

extern "C++" {

extern __device__ int abs(int a);
extern __device__ long int labs(long int a);
extern __device__ long long int llabs(long long int a);
extern __device__ unsigned long int max(const unsigned long int a,
                                        const long int b);
extern __device__ unsigned long int max(const long int a,
                                        const unsigned long int b);
extern __device__ unsigned long int max(const unsigned long int a,
                                        const unsigned long int b);
extern __device__ long int max(const long int a, const long int b);
extern __device__ unsigned int max(const unsigned int a, const int b);
extern __device__ unsigned int max(const int a, const unsigned int b);
extern __device__ unsigned int max(const unsigned int a, const unsigned int b);
extern __device__ int max(const int a, const int b);
extern __device__ unsigned long int min(const unsigned long int a,
                                        const long int b);
extern __device__ unsigned long int min(const long int a,
                                        const unsigned long int b);
extern __device__ unsigned long int min(const unsigned long int a,
                                        const unsigned long int b);
extern __device__ long int min(const long int a, const long int b);
extern __device__ unsigned int min(const unsigned int a, const int b);
extern __device__ unsigned int min(const int a, const unsigned int b);
extern __device__ unsigned int min(const unsigned int a, const unsigned int b);
extern __device__ int min(const int a, const int b);

extern __device__ long long int llmax(const long long int a,
                                      const long long int b);
extern __device__ long long int llmin(const long long int a,
                                      const long long int b);

extern __device__ unsigned int umax(const unsigned int a, const unsigned int b);
extern __device__ unsigned int umin(const unsigned int a, const unsigned int b);
extern __device__ unsigned long long int ullmax(const unsigned long long int a,
                                                const unsigned long long int b);
extern __device__ unsigned long long int ullmin(const unsigned long long int a,
                                                const unsigned long long int b);
}

#endif

#endif // include guard
