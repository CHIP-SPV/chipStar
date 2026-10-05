#!/bin/bash
# spirv-extractor --validate and -o must exit 0 on a valid fp64 module
# (chipStar issue #1770).
set -u
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
EXTRACTOR="@CMAKE_BINARY_DIR@/bin/spirv-extractor"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}" || exit 1
cat > k.hip <<'HIP'
#include <hip/hip_runtime.h>
__global__ void k(double *o, double a) { o[threadIdx.x] = a * 2.0; }
int main() { return 0; }
HIP
"${HIPCC}" k.hip -o k > build.log 2>&1 || { echo "FAIL: could not build"; tail -5 build.log; exit 1; }

"${EXTRACTOR}" --validate ./k > validate.log 2>&1 || { echo "FAIL: --validate exit $?"; cat validate.log; exit 1; }
"${EXTRACTOR}" -o k.spv ./k > dump.log 2>&1 || { echo "FAIL: -o exit $?"; cat dump.log; exit 1; }
echo "PASSED"
