#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1742. MEANT TO FAIL EVENTUALLY.
# Asserts the SPIR-V backend still cannot legalize a scalar to vector bitcast
# wider than a shader vector (GlobalISel fewerElementsBitcast). An LLVM without
# assertions may not abort; any successful compile counts as fixed.
# When it fires, replace the integer detour for wide accesses in
# HipVulkanRetypeArraysPass (llvm_passes/HipPasses.cpp, WORKAROUND #1742) with a
# vector bitcast and delete this test, then close #1742 naming the LLVM change.
set -u
LLC="@LLVM_TOOLS_BINARY_DIR@/llc"
cd "$(mktemp -d)"
cat > r.ll <<'IR'
target triple = "spirv-unknown-vulkan1.3-compute"
define internal i8 @f(double %d) {
  %v = bitcast double %d to <8 x i8>
  %e = extractelement <8 x i8> %v, i64 0
  ret i8 %e
}
define void @main() #0 {
  %r = call i8 @f(double 1.0)
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
IR
if "$LLC" -O0 -mtriple=spirv-unknown-vulkan1.3-compute r.ll -o r.spvt > llc.log 2>&1; then
  echo "CANARY FIRED: llc now compiles the bitcast, see the header"; exit 1
fi
grep -qE "fewerElementsBitcast|changeVectorElementCount" llc.log ||
  { echo "FAIL: llc failed for another reason"; cat llc.log; exit 1; }
echo "llc still fails:"; grep -m1 -E "Assertion|LLVM ERROR|error" llc.log
echo PASSED
