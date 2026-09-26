#!/bin/bash
# Regression test for CHIP-SPV/chipStar#1673: a call rewritten for dynamic
# shared memory must keep the calling convention and ABI attributes of the
# call it replaces, which LLVM requires to match the callee.
set -eu

SRC_DIR="@CMAKE_CURRENT_SOURCE_DIR@"
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_DIS="@LLVM_TOOLS_BINARY_DIR@/llvm-dis"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

# -O2 keeps h noinline, so the rewritten call is still in the lowered module.
# -fsigned-char: plain char is unsigned on aarch64 Linux, which gives zeroext.
"${HIPCC}" -O2 -fsigned-char --save-temps=cwd -c "${SRC_DIR}/@TEST_NAME@.hip" -o "@TEST_NAME@.o"
BC=$(ls ./*-lower.bc 2>/dev/null | head -1)
[ -n "${BC}" ] || { echo "FAIL: hipcc left no lowered device bitcode"; exit 1; }
"${LLVM_DIS}" "${BC}" -o lowered.ll

# h is spir_func and sign extends both its result and its argument.
CALLS=$(grep -v '^define' lowered.ll | grep '@_Z1hc(' || true)
GOOD=$(echo "${CALLS}" | grep -E 'call spir_func .*signext i8 @_Z1hc\(i8 .*signext ' || true)
if [ -z "${CALLS}" ] || [ "${CALLS}" != "${GOOD}" ]; then
  echo "FAIL: every call to h must be spir_func and sign extend, found:"
  echo "${CALLS}"
  exit 1
fi
echo "PASSED"
