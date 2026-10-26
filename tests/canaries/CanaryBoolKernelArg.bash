#!/bin/bash
# Canary for the i1 kernel-argument wrapper in llvm_passes/core/HipPromoteInts.cpp,
# https://github.com/CHIP-SPV/chipStar/issues/849 (no upstream report).
# MEANT TO FAIL EVENTUALLY. Asserts llvm-spirv and the in-tree SPIR-V backend still
# give a spir_kernel i1 parameter OpTypeBool, which the OpenCL SPIR-V environment
# spec (env/common_properties.asciidoc, Kernel Arguments) does not allow;
# spirv-val does not check it. Fires only once BOTH producers stop; then delete the
# kernel-arg wrapper code in llvm_passes/core/HipPromoteInts.cpp and this canary, and close #849.
set -u
cd "$(mktemp -d)"
printf '%s\n' 'target triple = "spirv64-unknown-unknown"' \
  'define spir_kernel void @k(ptr addrspace(1) %o, i1 zeroext %b) {' \
  '  %z = zext i1 %b to i32' '  store i32 %z, ptr addrspace(1) %o' '  ret void' '}' > m.ll
"@LLVM_TOOLS_BINARY_DIR@/llvm-as" m.ll -o m.bc || exit 1
bad=0
"@LLVM_SPIRV@" --spirv-text m.bc -o tr.spt > tr.log 2>&1 ||
  { echo "FAIL: llvm-spirv"; cat tr.log; exit 1; }
# Text form: "3 TypeBool <id>" then "3 FunctionParameter <type> <id>".
awk '$2 == "TypeBool" { b[$3] = 1 } $2 == "FunctionParameter" && b[$3] { f = 1 } END { exit !f }' tr.spt &&
  { echo "llvm-spirv: bool kernel parameter"; bad=1; }
"@LLVM_TOOLS_BINARY_DIR@/llc" -mtriple=spirv64v1.3-unknown-unknown m.ll -o be.spt > be.log 2>&1 ||
  { echo "FAIL: llc"; cat be.log; exit 1; }
awk '$3 == "OpTypeBool" { b[$1] = 1 } $3 == "OpFunctionParameter" && b[$4] { f = 1 } END { exit !f }' be.spt &&
  { echo "llc: bool kernel parameter"; bad=1; }
[ $bad = 1 ] && echo PASSED ||
  { echo "CANARY FIRED: no producer emits a bool kernel parameter, see the header"; exit 1; }
