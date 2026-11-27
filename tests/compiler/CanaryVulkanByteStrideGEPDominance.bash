#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/2398. MEANT TO FAIL EVENTUALLY.
# Asserts the SPIR-V backend still emits a GEP stepping through an [N x i8]
# stride before the GEP its base comes from. When it fires, delete
# HipVulkanTypeByteStrideGEPsPass (llvm_passes/HipPasses.cpp, WORKAROUND #2398)
# and this test together, then close #2398 naming the LLVM change that fixed it.
set -u
cd "$(mktemp -d)"
cat > r.ll <<'IR'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-chipstar-shadervulkan"
%struct.P = type { [20 x [5 x %struct.V]], [20 x float], i32 }
%struct.V = type { i32, float }
define spir_kernel void @k(i64 %i, ptr addrspace(1) %o) {
entry:
  %p = alloca %struct.P, align 4
  %fc = getelementptr i8, ptr %p, i64 800
  %ai = getelementptr [4 x i8], ptr %fc, i64 %i
  %v = load float, ptr %ai, align 4
  store float %v, ptr addrspace(1) %o, align 4
  ret void
}
IR
if "@LLVM_TOOLS_BINARY_DIR@/llc" -O3 -filetype=obj r.ll -o r.spv 2> err.txt; then
  echo "FAIL: the backend compiles the byte stride GEP; delete the #2398 workaround"
  exit 1
fi
grep -q "does not dominate" err.txt || { echo "FAIL: unexpected error"; cat err.txt; exit 1; }
echo PASSED
