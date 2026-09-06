// First translation unit of the reproducer for chipStar issue #1601.
// Holds only a single-precision kernel, so the device module built from this
// TU carries no OpTypeFloat 64. The double kernel lives in the second TU,
// inputs/ExtractorMultiTUDoublesTU2.hip, and the .bash links this one first so
// the doubles land in the SECOND __CLANG_OFFLOAD_BUNDLE__ of the fatbinary.
//
// spirv-extractor --check-for-doubles stopped at the first bundle, so it
// reported no doubles and let the test run on a device without fp64. Compiled
// by TestFix1601ExtractorMultiTUDoubles.bash; not a ctest itself (see
// RUNTIME_TESTS_MANUAL).
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void kFloat(float *P) { *P = 1.0f; }

int main() {
  // Never launched, and never meant to run at all: --check-for-doubles is
  // supposed to skip this binary before it starts. The ctest entry asserts the
  // wrapper's OUTPUT rather than its exit status, so reaching here fails that
  // entry by the absence of the skip marker, whatever this returns.
  printf("FAILED: this binary ran, so the fp64 kernel in the second "
         "translation unit was not detected\n");
  return 1;
}
