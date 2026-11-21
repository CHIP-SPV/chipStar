#!/bin/bash
# HipVerify must not cap llvm-spirv below a versioned triple:
# https://github.com/CHIP-SPV/chipStar/issues/2238
set -eu

OPT="@LLVM_TOOLS_BINARY_DIR@/opt"
PLUGIN="@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

cat > v.ll <<'IR'
target triple = "spirv64v1.3-unknown-chipstar"

define spir_kernel void @k(ptr addrspace(1) %p) {
  store i32 1, ptr addrspace(1) %p
  ret void
}
IR

CHIP_VERIFY_MODE=all "${OPT}" -load-pass-plugin "${PLUGIN}" \
  -passes=hip-post-link-passes v.ll -o /dev/null 2> v.txt
cat v.txt
! grep -q TripleMaxVersionIncompatible v.txt ||
  { echo "FAIL: llvm-spirv rejected the versioned triple"; exit 1; }
grep -Eq '\| *Post-HIP passes *\| *PASS *\| *PASS *\| *PASS *\|' v.txt ||
  { echo "FAIL: expected PASS PASS PASS in the Post-HIP passes row"; exit 1; }
echo PASSED
