#!/bin/bash
# Canary for HipFixOpenCLMDPass (llvm_passes/workarounds/HipFixOpenCLMD.*),
# https://github.com/CHIP-SPV/chipStar/issues/2101 (no upstream report). MEANT TO
# FAIL EVENTUALLY. Asserts both halves of
# the bug: clang still tags HIP device IR with opencl.ocl.version 0.0 (since
# https://github.com/llvm/llvm-project/commit/15a1769631ff0b2b3e830b03e51ae5f54f08a0ab),
# and llvm-spirv still aborts with "OCL version mismatch" when that meets the
# CL2.0 device library. When it fails, delete llvm_passes/workarounds/HipFixOpenCLMD.*, its registration and this
# canary, then close #2101.
set -u
cd "$(mktemp -d)"
printf '%s\n' '__attribute__((global)) void k(int *p) { *p = 1; }' > k.hip
"@LLVM_TOOLS_BINARY_DIR@/clang" -x hip --offload=spirv64 -nogpulib -nogpuinc \
  --offload-device-only -emit-llvm -S -O0 k.hip -o k.ll > m.log 2>&1 ||
  { echo "FAIL: clang did not compile"; cat m.log; exit 1; }
grep -q '^!opencl.ocl.version' k.ll && grep -q '= !{i32 0, i32 0}' k.ll ||
  { echo "CANARY FIRED: clang no longer emits opencl.ocl.version 0.0 for HIP"; grep ocl k.ll; exit 1; }
printf '%s\n' 'target triple = "spirv64-unknown-unknown"' \
  'define spir_kernel void @k() { ret void }' \
  '!opencl.ocl.version = !{!0, !1}' '!0 = !{i32 0, i32 0}' '!1 = !{i32 2, i32 0}' > m.ll
"@LLVM_TOOLS_BINARY_DIR@/llvm-as" m.ll -o m.bc || exit 1
if "@LLVM_SPIRV@" m.bc -o m.spv > t.log 2>&1; then
  echo "CANARY FIRED: llvm-spirv accepts mixed OpenCL versions, see the header"; exit 1
fi
grep -q 'OCL version mismatch' t.log || { echo "FAIL: unexpected error"; cat t.log; exit 1; }
echo PASSED
