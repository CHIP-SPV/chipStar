#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1654. MEANT TO FAIL
# EVENTUALLY: it asserts the in-tree SPIR-V backend still puts ContractionOff
# on a kernel whose FP ops all permit contraction. When it goes red, delete
# HipFPContractPass in llvm_passes/HipPasses.cpp, TestFix1654FPContract.bash
# and this file, then close #1654 naming the upstream change. The metadata
# default comes from https://github.com/llvm/llvm-project/pull/206404; no
# upstream issue exists.
set -eu

CLANG="@LLVM_TOOLS_BINARY_DIR@/clang"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

cat > canary.ll <<'EOF'
target triple = "spirv64"
define spir_kernel void @k(ptr %p, double %a, double %b) {
  %m = fmul contract double %a, %b
  %s = fadd contract double %m, %b
  store double %s, ptr %p
  ret void
}
EOF
"${CLANG}" -cc1 -triple spirv64v1.3-unknown-chipstar -emit-obj \
    canary.ll -o canary.spv

od -An -v -tx4 canary.spv > canary.words
# OpExecutionMode <id> ContractionOff is the words 0x00030010, <id>, 31.
if ! awk '{ for (i = 1; i <= NF; i++) { if (p2 && $i == "0000001f") f = 1
         p2 = p1; p1 = ($i == "00030010") } } END { exit !f }' canary.words; then
  echo "CANARY FIRED: the backend no longer forbids contraction for this kernel."
  echo "  HipFPContractPass is obsolete; see this file's header."
  exit 1
fi
echo "canary: backend still emits ContractionOff, the workaround is still needed"
echo "PASSED"
