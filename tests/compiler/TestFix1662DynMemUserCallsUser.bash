#!/bin/bash
# Regression test for CHIP-SPV/chipStar#1662: lowering a function that uses
# dynamic shared memory and calls another user of it must not crash, and both
# its own store and the store it reaches through the call must use the argument.
set -eu

SRC_DIR="@CMAKE_CURRENT_SOURCE_DIR@"
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_DIS="@LLVM_TOOLS_BINARY_DIR@/llvm-dis"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"
ARG="%smem__hidden_dyn_local_mem"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

# A use after free need not crash, so the lowered kernel is checked too.
"${HIPCC}" --save-temps=cwd -c "${SRC_DIR}/@TEST_NAME@.hip" -o "@TEST_NAME@.o"
BC=$(ls ./*-lower.bc 2>/dev/null | head -1)
[ -n "${BC}" ] || { echo "FAIL: hipcc left no lowered device bitcode"; exit 1; }
"${LLVM_DIS}" "${BC}" -o lowered.ll
awk '/^define .*@_Z1ki\(/ { f = 1 } f { print } f && /^}/ { f = 0 }' \
    lowered.ll > k.ll

# h is inlined by now, so its store shows up in k next to k's own.
STORES=$(grep -c "store i32 .*, ptr addrspace(3) ${ARG}" k.ll || true)
if [ "${STORES}" -ne 2 ]; then
  echo "FAIL: expected k's store and h's inlined store through ${ARG}, found ${STORES}"
  cat k.ll
  exit 1
fi
echo "PASSED"
