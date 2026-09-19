#!/bin/bash
# Regression test for CHIP-SPV/chipStar#1654: the in-tree SPIR-V backend must
# not forbid FP contraction in a module where llvm-spirv's default allows it.
set -eu

OPT="@LLVM_TOOLS_BINARY_DIR@/opt"
CLANG="@LLVM_TOOLS_BINARY_DIR@/clang"
PLUGIN="@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

# $1 = FP flags on the ops, $2 = extra call. Prints a one-kernel module.
kernel() {
  cat <<EOF
target triple = "spirv64"
declare double @ext_helper(double)
declare i64 @_Z12get_local_idj(i32)
define spir_func double @bad(double %x) noinline {
  %y = fadd double %x, %x
  ret double %y
}
define spir_kernel void @k(ptr %p, double %a, double %b) {
  %m = fmul $1 double %a, %b
  %s = fadd $1 double %m, %b
  $2
  store double %s, ptr %p
  ret void
}
EOF
}

# Exit 0 iff the od words in $1 hold OpExecutionMode <id> ContractionOff.
has_contraction_off() {
  awk '{ for (i = 1; i <= NF; i++) { if (p2 && $i == "0000001f") f = 1
         p2 = p1; p1 = ($i == "00030010") } } END { exit !f }' "$1"
}

# name, FP flags, extra call, expect ContractionOff (1) or not (0)
check() {
  kernel "$2" "$3" > "$1.ll"
  "${OPT}" -load-pass-plugin "${PLUGIN}" -passes=hip-post-link-passes \
      "$1.ll" -o "$1.bc"
  "${CLANG}" -cc1 -triple spirv64v1.3-unknown-chipstar -emit-obj \
      "$1.bc" -o "$1.spv"
  od -An -v -tx4 "$1.spv" > "$1.words"
  if has_contraction_off "$1.words"; then GOT=1; else GOT=0; fi
  if [ "${GOT}" -ne "$4" ]; then
    echo "FAIL: $1: expected ContractionOff=$4, got ${GOT}"
    exit 1
  fi
  echo "checked: $1 ContractionOff=${GOT}"
}

check contract contract "" 0
check uncontracted "" "" 1
check undefined-callee contract "%c = call double @ext_helper(double %s)" 1
check builtin-callee contract "%c = call i64 @_Z12get_local_idj(i32 0)" 0
check uncontracted-callee contract "%c = call double @bad(double %s)" 1

echo "PASSED"
