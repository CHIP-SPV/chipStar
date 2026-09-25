#!/bin/bash
# spirv-extractor must validate a device module of any SPIR-V version and write
# it out with -o unchanged (chipStar issue #1696).
set -u
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
EXTRACTOR="@CMAKE_BINARY_DIR@/bin/spirv-extractor"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}" || exit 1
cat > k.hip <<'EOF'
#include <hip/hip_runtime.h>
__global__ void k(float *o, float a) { o[threadIdx.x] = a * 2.0f; }
int main() { return 0; }
EOF
"${HIPCC}" k.hip -o k > build.log 2>&1 || { echo "FAIL: could not build"; tail -5 build.log; exit 1; }

if ! "${EXTRACTOR}" --validate ./k > validate.log 2>&1; then
  echo "FAIL: --validate rejected the module"; cat validate.log; exit 1
fi

"${EXTRACTOR}" -o k.spv ./k > dump.log 2>&1 || { echo "FAIL: -o failed"; cat dump.log; exit 1; }
# -o must write the embedded module itself, so its bytes occur in ./k verbatim.
hex() { od -An -v -tx1 "$1" | tr -d ' \n'; }
hex k.spv > spv.hex; hex k > k.hex
if [ ! -s spv.hex ] || ! grep -qF -f spv.hex k.hex; then
  echo "FAIL: -o wrote a module that is not the one embedded in ./k"; exit 1
fi
echo "PASSED"
