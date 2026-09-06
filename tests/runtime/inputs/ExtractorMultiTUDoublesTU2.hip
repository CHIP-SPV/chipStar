// Second translation unit for TestFix1601ExtractorMultiTUDoubles: the double
// kernel, and the only fp64 in the binary. Its device module becomes the
// second bundle of the fatbinary.
#include <hip/hip_runtime.h>

__global__ void kDouble(double *P) { *P = 2.0; }
