#include <hip/hip_runtime.h>

struct __align__(8) s0 {
  int a;
};
static_assert(alignof(s0) == 8, "Ineffective __align__ attribute!");

struct s1 {
  int a;
} __align__(16);
static_assert(alignof(s1) == 16, "Ineffective __align__ attribute!");

typedef struct __align__(32) s2 {
  int a;
} s2_t;
static_assert(alignof(s2_t) == 32, "Ineffective __align__ attribute!");

struct __align__(16) s3 {
  s2_t a;
};
static_assert(alignof(s3) == 32, "Ineffective __align__ attribute!");

__device__ s0 foo;
__device__ s1 bar;
__device__ s2_t baz;
__device__ s3 quz;
