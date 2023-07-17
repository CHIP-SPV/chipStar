// Regression test for https://github.com/CHIP-SPV/chipStar/issues/391
#include <hip/hip_runtime.h>
// NOTE: This format string has an invalid specifier which triggers
// undefined behavior. The goal with this test is to check the
// compiler does not crash.
__global__ void k() { printf("%\n"); }
