/*
 * Copyright (c) 2021-22 chipStar developers
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


#ifndef HIP_INCLUDE_DEVICELIB_INT_MATH_H
#define HIP_INCLUDE_DEVICELIB_INT_MATH_H

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
}

static inline __device__ long int labs(long int a) { return abs(a); }
static inline __device__ long long int abs(long long int a) { return abs((long int)a); }
static inline __device__ long long int llabs(long long int a) { return abs((long int)a); }

static inline __device__ unsigned long int max(const unsigned long int a, const long int b) {
  return (b < 0) ? a : max(a, (unsigned long)b);
}
static inline __device__ unsigned long int max(const long int a, const unsigned long int b) {
  return max(b, a);
}

static inline __device__ unsigned int max(const unsigned int a, const int b) {
  return (b < 0) ? a : max(a, (unsigned)b);
}
static inline __device__ unsigned int max(const int a, const unsigned int b) {
  return max(b, a);
}

static inline __device__ unsigned long int min(const unsigned long int a, const long int b) {
  return (b < 0) ? 0 : min(a, (unsigned long)b);
}
static inline __device__ unsigned long int min(const long int a, const unsigned long int b) {
  return min(b, a);
}

static inline __device__ unsigned int min(const unsigned int a, const int b) {
  return (b < 0) ? 0 : min(a, (unsigned)b);
}
static inline __device__ unsigned int min(const int a, const unsigned int b) {
  return min(b, a);
}

static inline __device__ unsigned int umax(const unsigned int a, const unsigned int b) {
  return max(a, b);
}

static inline __device__ unsigned int umin(const unsigned int a, const unsigned int b) {
  return min(a, b);
}

static inline __device__ unsigned long long int max(unsigned long long int a,
                                      unsigned long long int b) {
  return max((unsigned long)a, (unsigned long)b);
}

static inline __device__ unsigned long long int min(unsigned long long int a,
                                      unsigned long long int b) {
  return min((unsigned long)a, (unsigned long)b);
}

static inline __device__ unsigned long long int ullmax(const unsigned long long int a,
                                         const unsigned long long int b) {
  return max((unsigned long)a, (unsigned long)b);
}

static inline __device__ unsigned long long int ullmin(const unsigned long long int a,
                                         const unsigned long long int b) {
  return min((unsigned long)a, (unsigned long)b);
}

static inline __device__ long long int llmax(const long long int a, const long long int b) {
  return max((long)a, (long)b);
}

static inline __device__ long long int llmin(const long long int a, const long long int b) {
  return min((long)a, (long)b);
}

namespace std {
// Clang does provide device side std::abs via HIP include wrappers
// but, alas, the wrappers won't compile on chipStar due to AMD
// specific built-ins.
using ::abs;
} // namespace std

#endif // include guard
