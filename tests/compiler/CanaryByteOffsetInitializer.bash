#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1703. MEANT TO FAIL
# EVENTUALLY: it asserts the in-tree SPIR-V backend still emits the byte offset
# of getelementptr (i8, @D, 12) in a global initializer as an array index over
# @D's own type, which addresses &D[12] instead of &D[3]. When it goes red,
# delete retypeToBytes and its call in HipOffsetBeforeCastPass
# (llvm_passes/HipPasses.cpp) and this file, and close #1703. No upstream
# report names this; the reproducer attached to
# https://github.com/llvm/llvm-project/issues/95760 shows it.
set -eu

LLC="@LLVM_TOOLS_BINARY_DIR@/llc"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

printf '%s\n' 'target triple = "spirv64-unknown-unknown"' \
  '@D = internal addrspace(1) constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]' \
  '@T = addrspace(1) global ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @D, i64 12)' \
  'define spir_kernel void @k(ptr addrspace(1) %o) {' \
  '  %p = load ptr addrspace(1), ptr addrspace(1) @T' \
  '  store ptr addrspace(1) %p, ptr addrspace(1) %o' \
  '  ret void' '}' > m.ll
"${LLC}" -mtriple=spirv64-unknown-unknown m.ll -o m.s > m.log 2>&1 ||
  { echo "FAIL: llc does not compile the module"; cat m.log; exit 1; }

D=$(awk '$1 == "OpName" && $3 == "\"D\"" { print $2 }' m.s)
C=$(awk '$3 == "OpConstant" && $5 == "12" { printf "%s%s", s, $1; s="|" }' m.s)
# The id of @D's pointer type, of its pointee, and of that array's element.
P=$(awk -v d="${D}" '$1 == d && $3 == "OpVariable" { print $4 }' m.s)
A=$(awk -v p="${P}" '$1 == p && $3 == "OpTypePointer" { print $5 }' m.s)
E=$(awk -v a="${A}" '$1 == a && $3 == "OpTypeArray" { print $4 }' m.s)
[ -n "${E}" ] ||
  { echo "FAIL: cannot find @D's array type"; cat m.s; exit 1; }
if ! grep -qE "^[[:space:]]*${E} = OpTypeInt 32 " m.s ||
   ! grep -qE "OpSpecConstantOp %[0-9]+ (InBounds)?PtrAccessChain ${D} %[0-9]+ (${C:-none})\$" m.s; then
  echo "CANARY FIRED: the SPIR-V backend no longer indexes an int array @D by"
  echo "  the byte offset in an initializer. Read this file's header."
  grep 'OpSpecConstantOp' m.s
  exit 1
fi
echo "canary: the backend still indexes @D by bytes, the workaround is still needed"
echo PASSED
