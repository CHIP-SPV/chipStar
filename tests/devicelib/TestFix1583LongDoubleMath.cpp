// Reproduces CHIP-SPV/chipStar#1583: chipStar's devicelib declares only
// double and __half overloads of the math functions it hoists into
// namespace std, so device-side code that calls them with 'long double' -
// or calls unqualified abs() on a floating-point value - is ambiguous.
// Compile-only test: it never has to run, it only has to compile.
#include <hip/hip_runtime.h>
#include <cmath>

__device__ long double rsqrtRef(long double Arg) { return 1.0L / std::sqrt(Arg); }

__device__ long double powRef(long double A, long double B) {
  return std::pow(A, B);
}

// boost::math::ccmath::abs<double> calls unqualified abs() like this.
__device__ double absDouble(double Arg) { return abs(Arg); }
__device__ float absFloat(float Arg) { return abs(Arg); }
__device__ long double absLongDouble(long double Arg) { return abs(Arg); }

// Integer arguments must keep picking the plain double overloads: a plain
// long double overload would tie with the double one here, since int ->
// double and int -> long double are the same conversion rank.
__device__ double powIntShort(int A, short B) { return std::pow(A, B); }

__global__ void k(double *Out) {
  *Out = static_cast<double>(rsqrtRef(4.0L)) +
         static_cast<double>(powRef(2.0L, 3.0L)) + absDouble(-3.0) +
         absFloat(-3.0f) + static_cast<double>(absLongDouble(-3.0L)) +
         powIntShort(2, 3);
}

int main() { return 0; }
