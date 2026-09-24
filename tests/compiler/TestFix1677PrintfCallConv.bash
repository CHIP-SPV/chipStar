#!/bin/bash
# Regression test for CHIP-SPV/chipStar#1677: every call HipPrintf builds must
# use the calling convention of its callee, which LLVM requires to match.
set -eu

SRC_DIR="@CMAKE_CURRENT_SOURCE_DIR@"
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_DIS="@LLVM_TOOLS_BINARY_DIR@/llvm-dis"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

"${HIPCC}" -Wno-format --save-temps=cwd -c "${SRC_DIR}/@TEST_NAME@.hip" \
  -o "@TEST_NAME@.o"
BC=$(ls ./*-lower.bc 2>/dev/null | head -1)
[ -n "${BC}" ] || { echo "FAIL: hipcc left no lowered device bitcode"; exit 1; }
"${LLVM_DIS}" "${BC}" -o lowered.ll

CALLS=$(grep -v '^define' lowered.ll | grep -E '@(printf|_cl_print_str)\(' |
  grep -E 'call ' || true)
GOOD=$(echo "${CALLS}" | grep 'call spir_func ' || true)
# Both lowered printf shapes and _cl_print_str must be present.
for F in printf _cl_print_str; do
  echo "${CALLS}" | grep -q "@${F}(" || { echo "FAIL: no call to ${F}"; exit 1; }
done
if [ "${CALLS}" != "${GOOD}" ]; then
  echo "FAIL: every printf and _cl_print_str call must be spir_func, found:"
  echo "${CALLS}"
  exit 1
fi
echo "PASSED"
