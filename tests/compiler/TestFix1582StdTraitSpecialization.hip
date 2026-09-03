// Reproduces issue #1582: hip_fp16.h must not specialize std::is_floating_point
// for _Float16. That is UB ([namespace.std] allows specializing a standard
// template only for program-defined types) and libc++ >= 20 marks the trait
// [[clang::no_specializations]], turning it into a hard error.
#include <hip/hip_fp16.h>
#include <type_traits>

// The standard trait must be left exactly as the standard library defines it.
static_assert(!std::is_floating_point<_Float16>::value,
              "chipStar must not specialize std::is_floating_point<_Float16>");

// The __half conversions that the specialization used to enable must keep
// working through a chipStar-local trait.
__host__ __device__ void half_conversions() {
    _Float16 f16 = 1.5;
    __half h{f16};
    h = f16;
    _Float16 back = h;

    __half hf{1.5f};
    hf = 2.5f;
    float f = hf;

    __half hd{1.5};
    hd = 2.5;
    double d = hd;

    (void)back; (void)f; (void)d;
}

int main() { return 0; }
