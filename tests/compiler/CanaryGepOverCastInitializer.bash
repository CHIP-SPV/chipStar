#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1693. MEANT TO FAIL
# EVENTUALLY: it asserts the in-tree SPIR-V backend still aborts on a
# getelementptr over an addrspacecast in a global initializer, the shape
# HipOffsetBeforeCastPass in llvm_passes/HipPasses.cpp rewrites. When it goes
# red, delete this file and say so on #1693; close #1693 only once its
# ptrtoint form emits valid SPIR-V too. Delete the pass and
# tests/runtime/TestFix1693StringTableOffset.hip only once IGC also resolves
# the original shape instead of storing 0
# (https://github.com/CHIP-SPV/chipStar/issues/1695) and
# CanaryByteOffsetInitializer has fired too; this canary does not check IGC.
# No upstream report exists for either bug.
set -eu

CLANG="@LLVM_TOOLS_BINARY_DIR@/clang"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

emit() { # $1: the initializer of @P
  printf '%s\n' 'target triple = "spirv64"' \
    '@S = private addrspace(1) constant [4 x i8] c"abc\00"' \
    "@P = addrspace(1) constant ptr addrspace(4) $1" \
    'define spir_kernel void @k(ptr addrspace(1) %o) {' \
    '  %v = load ptr addrspace(4), ptr addrspace(1) @P' \
    '  store ptr addrspace(4) %v, ptr addrspace(1) %o' \
    '  ret void' '}' > m.ll
  "${CLANG}" -cc1 -triple spirv64v1.3-unknown-chipstar -emit-obj m.ll \
    -o m.spv > m.log 2>&1
}

# Control: the shape the pass emits must compile.
emit 'addrspacecast (ptr addrspace(1) getelementptr (i8, ptr addrspace(1) @S, i64 1) to ptr addrspace(4))' ||
  { echo "FAIL: the control module does not compile"; cat m.log; exit 1; }

echo "canary: an abort below is expected today"
if emit 'getelementptr (i8, ptr addrspace(4) addrspacecast (ptr addrspace(1) @S to ptr addrspace(4)), i64 1)'; then
  echo "CANARY FIRED: the SPIR-V backend now compiles a getelementptr over an"
  echo "  addrspacecast in an initializer. Read this file's header."
  exit 1
fi
grep -qF 'getImm(I.getOperand(2), MRI)' m.log ||
  { echo "FAIL: rejected for another reason"; head -5 m.log; exit 1; }
echo "canary: the backend still aborts, the workaround is still needed"
echo PASSED
