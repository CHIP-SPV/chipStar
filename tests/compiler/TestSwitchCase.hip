// Regression test for optimized switch case with non-standard label
// select bitwidth (e.g. i4) which makes llvm-spirv crash.
// Related issue: https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/1685
#include <hip/hip_runtime.h>

__global__ void k(float *Data, int Sel) {
  switch (Sel & 0xf) {
  case 0:
    *Data += 1.23f;
    break;
  case 3:
    *Data += 12.3f;
    break;
  }
}
