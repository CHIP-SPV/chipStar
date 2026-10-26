#!/bin/bash
# Canary for HipLowerZeroLengthArraysPass (llvm_passes/workarounds/HipLowerZeroLengthArrays.cpp),
# https://github.com/CHIP-SPV/chipStar/issues/2109 (close it when deleting).
# Register only for LLVM < 22 with the in-tree SPIR-V backend: LLVM 22 legalizes
# zero-size arrays (https://github.com/llvm/llvm-project/pull/172367) and the
# translator maps them to [1 x T] since
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/2743. MEANT TO FAIL
# EVENTUALLY. Asserts llc still emits OpTypeRuntimeArray (rejected by IGC with
# "Unimplemented opcode 29") for [0 x float] in a Kernel module. When LLVM 21
# support is dropped, delete the pass and this canary.
set -u
cd "$(mktemp -d)"
printf '%s\n' 'target triple = "spirv64-unknown-unknown"' \
  '@a = addrspace(1) global [3 x float] zeroinitializer' \
  'define spir_kernel void @k(ptr addrspace(1) %o, i64 %i) {' \
  '  %p = getelementptr inbounds [0 x float], ptr addrspace(1) @a, i64 0, i64 %i' \
  '  %v = load float, ptr addrspace(1) %p' '  store float %v, ptr addrspace(1) %o' \
  '  ret void' '}' > m.ll
"@LLVM_TOOLS_BINARY_DIR@/llc" -mtriple=spirv64v1.3-unknown-unknown m.ll -o m.spt > m.log 2>&1 ||
  { echo "CANARY FIRED: llc rejects the module (LLVM 22+ behavior), see the header"; cat m.log; exit 1; }
grep -q OpTypeRuntimeArray m.spt && echo PASSED ||
  { echo "CANARY FIRED: llc no longer emits OpTypeRuntimeArray, see the header"; exit 1; }
