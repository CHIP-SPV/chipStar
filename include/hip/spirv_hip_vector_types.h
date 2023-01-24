/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

/**
 *  @file  hcc_detail/hip_vector_types.h
 *  @brief Defines the different newt vector types for HIP runtime.
 */

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_VECTOR_TYPES_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_VECTOR_TYPES_H

#if defined(__cplusplus) && defined(__has_attribute) &&                        \
    __has_attribute(ext_vector_type)
#define __NATIVE_VECTOR__(n, T) T __attribute__((ext_vector_type(n)))
#else
#define __NATIVE_VECTOR__(n, T) T[n]
#endif

#if defined(__cplusplus)
#include <type_traits>

template <typename T, unsigned int n>
struct HIP_vector_base;

template <typename T>
struct HIP_vector_base<T, 1> {
  // SPIR-V does not have one element vector types and the current
  // Clang -> SPIR-V translator tool chain used by CHIP-SPV does not
  // include a legalization phase to lower one element vectors to
  // scalars. Therefore, define Native_vec_ to be a scalar here.
  //
  // When the SPIR-V backend lands on the LLVM we may reintroduce one
  // element vectors.
  using Native_vec_ = T /*__NATIVE_VECTOR__(1, T) */;

  union {
    Native_vec_ data;
    T array[1];
    struct {
      T x;
    };
  };
};

template <typename T>
struct HIP_vector_base<T, 2> {
  using Native_vec_ = __NATIVE_VECTOR__(2, T);

  union {
    Native_vec_ data;
    T array[2];
    struct {
      T x;
      T y;
    };
  };
};

template <typename T>
struct HIP_vector_base<T, 3> {
  using Native_vec_ = __NATIVE_VECTOR__(3, T);

  union {
    Native_vec_ data;
    T array[3];
    struct {
      T x;
      T y;
      T z;
    };
  };
};

template <typename T>
struct HIP_vector_base<T, 4> {
  using Native_vec_ = __NATIVE_VECTOR__(4, T);

  union {
    Native_vec_ data;
    T array[4];
    struct {
      T x;
      T y;
      T z;
      T w;
    };
  };
};

template <typename T, unsigned int rank>
struct HIP_vector_type : public HIP_vector_base<T, rank> {
  using HIP_vector_base<T, rank>::data;
  using HIP_vector_base<T, rank>::array;
  using typename HIP_vector_base<T, rank>::Native_vec_;

  __host__ __device__ HIP_vector_type() = default;
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>{}>::type* = nullptr>
  __host__ __device__ HIP_vector_type(U x) noexcept {
    for (auto i = 0u; i != rank; ++i) array[i] = x;
  }
  template <  // TODO: constrain based on type as well.
      typename... Us, typename std::enable_if<
                          (rank > 1) && sizeof...(Us) == rank>::type* = nullptr>
  __host__ __device__ HIP_vector_type(Us... xs) noexcept {
    data = Native_vec_{static_cast<T>(xs)...};
  }
  __host__ __device__ HIP_vector_type(const HIP_vector_type&) = default;
  __host__ __device__ HIP_vector_type(HIP_vector_type&&) = default;
  __host__ __device__ ~HIP_vector_type() = default;

  __host__ __device__ HIP_vector_type& operator=(const HIP_vector_type&) =
      default;
  __host__ __device__ HIP_vector_type& operator=(HIP_vector_type&&) = default;

  // Operators
  __host__ __device__ HIP_vector_type& operator++() noexcept {
    return *this += HIP_vector_type{1};
  }
  __host__ __device__ HIP_vector_type operator++(int) noexcept {
    auto tmp(*this);
    ++*this;
    return tmp;
  }
  __host__ __device__ HIP_vector_type& operator--() noexcept {
    return *this -= HIP_vector_type{1};
  }
  __host__ __device__ HIP_vector_type operator--(int) noexcept {
    auto tmp(*this);
    --*this;
    return tmp;
  }
  __host__ __device__ HIP_vector_type& operator+=(
      const HIP_vector_type& x) noexcept {
    data += x.data;
    return *this;
  }
  __host__ __device__ HIP_vector_type& operator-=(
      const HIP_vector_type& x) noexcept {
    data -= x.data;
    return *this;
  }
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>{}>::type* = nullptr>
  __host__ __device__ HIP_vector_type& operator-=(U x) noexcept {
    return *this -= HIP_vector_type{x};
  }
  __host__ __device__ HIP_vector_type& operator*=(
      const HIP_vector_type& x) noexcept {
    data *= x.data;
    return *this;
  }
  __host__ __device__ HIP_vector_type& operator/=(
      const HIP_vector_type& x) noexcept {
    data /= x.data;
    return *this;
  }

  template <typename U = T,
            typename std::enable_if<std::is_signed<U>{}>::type* = nullptr>
  __host__ __device__ HIP_vector_type operator-() noexcept {
    auto tmp(*this);
    tmp.data = -tmp.data;
    return tmp;
  }

  template <typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
  __host__ __device__ HIP_vector_type operator~() noexcept {
    HIP_vector_type r{*this};
    r.data = ~r.data;
    return r;
  }
  template <typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
  __host__ __device__ HIP_vector_type& operator%=(
      const HIP_vector_type& x) noexcept {
    data %= x.data;
    return *this;
  }
  template <typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
  __host__ __device__ HIP_vector_type& operator^=(
      const HIP_vector_type& x) noexcept {
    data ^= x.data;
    return *this;
  }
  template <typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
  __host__ __device__ HIP_vector_type& operator|=(
      const HIP_vector_type& x) noexcept {
    data |= x.data;
    return *this;
  }
  template <typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
  __host__ __device__ HIP_vector_type& operator&=(
      const HIP_vector_type& x) noexcept {
    data &= x.data;
    return *this;
  }
  template <typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
  __host__ __device__ HIP_vector_type& operator>>=(
      const HIP_vector_type& x) noexcept {
    data >>= x.data;
    return *this;
  }
  template <typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
  __host__ __device__ HIP_vector_type& operator<<=(
      const HIP_vector_type& x) noexcept {
    data <<= x.data;
    return *this;
  }
};

template <typename T, unsigned int n>
__host__ __device__ inline HIP_vector_type<T, n> operator+(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} += y;
}
template <typename T, unsigned int n, typename U>
__host__ __device__ inline HIP_vector_type<T, n> operator+(
    const HIP_vector_type<T, n>& x, U y) noexcept {
  return HIP_vector_type<T, n>{x} += y;
}
template <typename T, unsigned int n, typename U>
__host__ __device__ inline HIP_vector_type<T, n> operator+(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return y + x;
}

template <typename T, unsigned int n>
__host__ __device__ inline HIP_vector_type<T, n> operator-(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} -= y;
}
template <typename T, unsigned int n, typename U>
__host__ __device__ inline HIP_vector_type<T, n> operator-(
    const HIP_vector_type<T, n>& x, U y) noexcept {
  return HIP_vector_type<T, n>{x} -= y;
}
template <typename T, unsigned int n, typename U>
__host__ __device__ inline HIP_vector_type<T, n> operator-(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} -= y;
}

template <typename T, unsigned int n>
__host__ __device__ inline HIP_vector_type<T, n> operator*(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} *= y;
}
template <typename T, unsigned int n, typename U>
__host__ __device__ inline HIP_vector_type<T, n> operator*(
    const HIP_vector_type<T, n>& x, U y) noexcept {
  return HIP_vector_type<T, n>{x} *= y;
}
template <typename T, unsigned int n, typename U>
__host__ __device__ inline HIP_vector_type<T, n> operator*(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return y * x;
}

template <typename T, unsigned int n>
__host__ __device__ inline HIP_vector_type<T, n> operator/(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} /= y;
}
template <typename T, unsigned int n, typename U>
__host__ __device__ inline HIP_vector_type<T, n> operator/(
    const HIP_vector_type<T, n>& x, U y) noexcept {
  return HIP_vector_type<T, n>{x} /= y;
}
template <typename T, unsigned int n, typename U>
__host__ __device__ inline HIP_vector_type<T, n> operator/(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} /= y;
}

template <typename T, unsigned int n>
__host__ __device__ inline bool operator==(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  // Original code:
  // auto tmp = x.data == y.data;
  // for (auto i = 0u; i != n; ++i)
  //   if (tmp[i] == 0) return false;
  // return true;
  for (auto i = 0u; i != n; ++i)
    if (x.array[i] != y.array[i])
      return false;
  return true;
}
template <typename T, unsigned int n, typename U>
__host__ __device__ inline bool operator==(const HIP_vector_type<T, n>& x,
                                           U y) noexcept {
  return x == HIP_vector_type<T, n>{y};
}
template <typename T, unsigned int n, typename U>
__host__ __device__ inline bool operator==(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} == y;
}

template <typename T, unsigned int n>
__host__ __device__ inline bool operator!=(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return !(x == y);
}
template <typename T, unsigned int n, typename U>
__host__ __device__ inline bool operator!=(const HIP_vector_type<T, n>& x,
                                           U y) noexcept {
  return !(x == y);
}
template <typename T, unsigned int n, typename U>
__host__ __device__ inline bool operator!=(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return !(x == y);
}

template <typename T, unsigned int n,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator%(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} %= y;
}
template <typename T, unsigned int n, typename U,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator%(const HIP_vector_type<T, n>& x,
                                       U y) noexcept {
  return HIP_vector_type<T, n>{x} %= y;
}
template <typename T, unsigned int n, typename U,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator%(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} %= y;
}

template <typename T, unsigned int n,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator^(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} ^= y;
}
template <typename T, unsigned int n, typename U,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator^(const HIP_vector_type<T, n>& x,
                                       U y) noexcept {
  return HIP_vector_type<T, n>{x} ^= y;
}
template <typename T, unsigned int n, typename U,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator^(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} ^= y;
}

template <typename T, unsigned int n,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator|(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} |= y;
}
template <typename T, unsigned int n, typename U,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator|(const HIP_vector_type<T, n>& x,
                                       U y) noexcept {
  return HIP_vector_type<T, n>{x} |= y;
}
template <typename T, unsigned int n, typename U,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator|(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} |= y;
}

template <typename T, unsigned int n,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator&(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} &= y;
}
template <typename T, unsigned int n, typename U,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator&(const HIP_vector_type<T, n>& x,
                                       U y) noexcept {
  return HIP_vector_type<T, n>{x} &= y;
}
template <typename T, unsigned int n, typename U,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator&(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} &= y;
}

template <typename T, unsigned int n,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator>>(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} >>= y;
}
template <typename T, unsigned int n, typename U,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator>>(const HIP_vector_type<T, n>& x,
                                        U y) noexcept {
  return HIP_vector_type<T, n>{x} >>= y;
}
template <typename T, unsigned int n, typename U,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator>>(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} >>= y;
}

template <typename T, unsigned int n,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator<<(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} <<= y;
}
template <typename T, unsigned int n, typename U,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator<<(const HIP_vector_type<T, n>& x,
                                        U y) noexcept {
  return HIP_vector_type<T, n>{x} <<= y;
}
template <typename T, unsigned int n, typename U,
          typename std::enable_if<std::is_integral<T>{}>* = nullptr>
inline HIP_vector_type<T, n> operator<<(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} <<= y;
}

#define __MAKE_VECTOR_TYPE__(CUDA_name, T)    \
  using CUDA_name##1 = HIP_vector_type<T, 1>; \
  using CUDA_name##2 = HIP_vector_type<T, 2>; \
  using CUDA_name##3 = HIP_vector_type<T, 3>; \
  using CUDA_name##4 = HIP_vector_type<T, 4>;

#else  // __cplusplus

#define __MAKE_VECTOR_TYPE__(CUDA_name, T)                                     \
  typedef struct {                                                             \
    T x;                                                                       \
  } CUDA_name##1;                                                              \
  typedef struct {                                                             \
    T x;                                                                       \
    T y;                                                                       \
  } CUDA_name##2;                                                              \
  typedef struct {                                                             \
    T x;                                                                       \
    T y;                                                                       \
    T z;                                                                       \
  } CUDA_name##3;                                                              \
  typedef struct {                                                             \
    T x;                                                                       \
    T y;                                                                       \
    T z;                                                                       \
    T w;                                                                       \
  } CUDA_name##4;

/*
#define __MAKE_VECTOR_TYPE__(CUDA_name, T) \
typedef T CUDA_name##1 __NATIVE_VECTOR__(1, T);\
typedef T CUDA_name##2 __NATIVE_VECTOR__(2, T);\
typedef T CUDA_name##3 __NATIVE_VECTOR__(3, T);\
typedef T CUDA_name##4 __NATIVE_VECTOR__(4, T);
#endif
*/

#endif  // __cplusplus

__MAKE_VECTOR_TYPE__(uchar, unsigned char);
__MAKE_VECTOR_TYPE__(char, char);
__MAKE_VECTOR_TYPE__(ushort, unsigned short);
__MAKE_VECTOR_TYPE__(short, short);
__MAKE_VECTOR_TYPE__(uint, unsigned int);
__MAKE_VECTOR_TYPE__(int, int);
__MAKE_VECTOR_TYPE__(ulong, unsigned long);
__MAKE_VECTOR_TYPE__(long, long);
__MAKE_VECTOR_TYPE__(ulonglong, unsigned long long);
__MAKE_VECTOR_TYPE__(longlong, long long);
__MAKE_VECTOR_TYPE__(float, float);
__MAKE_VECTOR_TYPE__(double, double);

/*
#define DECLOP_MAKE_ONE_COMPONENT(comp, type) \
    __device__ __host__ \
    static \
    inline \
    type make_##type(comp x) {  \
      type r; r.data = (type##_impl)(x); \
      return r; \
    }

#define DECLOP_MAKE_TWO_COMPONENT(comp, type) \
    __device__ __host__ \
    static \
    inline \
    type make_##type(comp x, comp y) { \
      type r; r.data = (type##_impl)(x, y); \
      return r; \
    }

#define DECLOP_MAKE_THREE_COMPONENT(comp, type) \
    __device__ __host__ \
    static \
    inline \
    type make_##type(comp x, comp y, comp z) { \
      type r; r.data = (type##_impl)(x, y, z); \
      return r; \
    }

#define DECLOP_MAKE_FOUR_COMPONENT(comp, type) \
    __device__ __host__ \
    static \
    inline \
    type make_##type(comp x, comp y, comp z, comp w) { \
        type r; r.data = (type##_impl)(x, y, z, w); \
        return r; \
    }
*/

#define DECLOP_MAKE_ONE_COMPONENT(comp, type)                  \
  __device__ __host__ static inline type make_##type(comp x) { \
    type r;                                                    \
    r.x = x;                                                   \
    return r;                                                  \
  }

#define DECLOP_MAKE_TWO_COMPONENT(comp, type)                          \
  __device__ __host__ static inline type make_##type(comp x, comp y) { \
    type r;                                                            \
    r.x = x;                                                           \
    r.y = y;                                                           \
    return r;                                                          \
  }

#define DECLOP_MAKE_THREE_COMPONENT(comp, type)                                \
  __device__ __host__ static inline type make_##type(comp x, comp y, comp z) { \
    type r;                                                                    \
    r.x = x;                                                                   \
    r.y = y;                                                                   \
    r.z = z;                                                                   \
    return r;                                                                  \
  }

#define DECLOP_MAKE_FOUR_COMPONENT(comp, type)                               \
  __device__ __host__ static inline type make_##type(comp x, comp y, comp z, \
                                                     comp w) {               \
    type r;                                                                  \
    r.x = x;                                                                 \
    r.y = y;                                                                 \
    r.z = z;                                                                 \
    r.w = w;                                                                 \
    return r;                                                                \
  }

DECLOP_MAKE_ONE_COMPONENT(unsigned char, uchar1);
DECLOP_MAKE_TWO_COMPONENT(unsigned char, uchar2);
DECLOP_MAKE_THREE_COMPONENT(unsigned char, uchar3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned char, uchar4);

DECLOP_MAKE_ONE_COMPONENT(signed char, char1);
DECLOP_MAKE_TWO_COMPONENT(signed char, char2);
DECLOP_MAKE_THREE_COMPONENT(signed char, char3);
DECLOP_MAKE_FOUR_COMPONENT(signed char, char4);

DECLOP_MAKE_ONE_COMPONENT(unsigned short, ushort1);
DECLOP_MAKE_TWO_COMPONENT(unsigned short, ushort2);
DECLOP_MAKE_THREE_COMPONENT(unsigned short, ushort3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned short, ushort4);

DECLOP_MAKE_ONE_COMPONENT(signed short, short1);
DECLOP_MAKE_TWO_COMPONENT(signed short, short2);
DECLOP_MAKE_THREE_COMPONENT(signed short, short3);
DECLOP_MAKE_FOUR_COMPONENT(signed short, short4);

DECLOP_MAKE_ONE_COMPONENT(unsigned int, uint1);
DECLOP_MAKE_TWO_COMPONENT(unsigned int, uint2);
DECLOP_MAKE_THREE_COMPONENT(unsigned int, uint3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned int, uint4);

DECLOP_MAKE_ONE_COMPONENT(signed int, int1);
DECLOP_MAKE_TWO_COMPONENT(signed int, int2);
DECLOP_MAKE_THREE_COMPONENT(signed int, int3);
DECLOP_MAKE_FOUR_COMPONENT(signed int, int4);

DECLOP_MAKE_ONE_COMPONENT(float, float1);
DECLOP_MAKE_TWO_COMPONENT(float, float2);
DECLOP_MAKE_THREE_COMPONENT(float, float3);
DECLOP_MAKE_FOUR_COMPONENT(float, float4);

DECLOP_MAKE_ONE_COMPONENT(double, double1);
DECLOP_MAKE_TWO_COMPONENT(double, double2);
DECLOP_MAKE_THREE_COMPONENT(double, double3);
DECLOP_MAKE_FOUR_COMPONENT(double, double4);

DECLOP_MAKE_ONE_COMPONENT(unsigned long, ulong1);
DECLOP_MAKE_TWO_COMPONENT(unsigned long, ulong2);
DECLOP_MAKE_THREE_COMPONENT(unsigned long, ulong3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned long, ulong4);

DECLOP_MAKE_ONE_COMPONENT(signed long, long1);
DECLOP_MAKE_TWO_COMPONENT(signed long, long2);
DECLOP_MAKE_THREE_COMPONENT(signed long, long3);
DECLOP_MAKE_FOUR_COMPONENT(signed long, long4);

DECLOP_MAKE_ONE_COMPONENT(unsigned long long, ulonglong1);
DECLOP_MAKE_TWO_COMPONENT(unsigned long long, ulonglong2);
DECLOP_MAKE_THREE_COMPONENT(unsigned long long, ulonglong3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned long long, ulonglong4);

DECLOP_MAKE_ONE_COMPONENT(signed long long, longlong1);
DECLOP_MAKE_TWO_COMPONENT(signed long long, longlong2);
DECLOP_MAKE_THREE_COMPONENT(signed long long, longlong3);
DECLOP_MAKE_FOUR_COMPONENT(signed long long, longlong4);

#endif
