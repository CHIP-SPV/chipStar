#!/bin/bash
# Regression test for CHIP-SPV/chipStar#1670: every call rewritten for dynamic
# shared memory must pass its caller's argument, whichever of the two functions
# is cloned first.
set -eu

SRC_DIR="@CMAKE_CURRENT_SOURCE_DIR@"
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_DIS="@LLVM_TOOLS_BINARY_DIR@/llvm-dis"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"
ARG="%smem__hidden_dyn_local_mem"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

# A wrong lowering need not crash, so the lowered kernels are checked too.
"${HIPCC}" --save-temps=cwd -c "${SRC_DIR}/@TEST_NAME@.hip" -o "@TEST_NAME@.o"
BC=$(ls ./*-lower.bc 2>/dev/null | head -1)
[ -n "${BC}" ] || { echo "FAIL: hipcc left no lowered device bitcode"; exit 1; }
"${LLVM_DIS}" "${BC}" -o lowered.ll

# b, c and h are inlined by now: each kernel stores through its own argument
# once directly and once through the other user.
for K in _Z1ai _Z1ki; do
  awk "/^define .*@${K}\\(/ { f = 1 } f { print } f && /^}/ { f = 0 }" \
      lowered.ll > "${K}.ll"
  STORES=$(grep -c "store i32 .*, ptr addrspace(3) ${ARG}" "${K}.ll" || true)
  if [ "${STORES}" -ne 2 ]; then
    echo "FAIL: expected 2 stores through ${ARG} in ${K}, found ${STORES}"
    cat "${K}.ll"
    exit 1
  fi
done
echo "PASSED"
