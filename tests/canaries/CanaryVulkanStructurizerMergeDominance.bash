#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1738. MEANT TO FAIL EVENTUALLY.
# Asserts the SPIR-V backend still emits a selection construct whose header does
# not dominate its merge block when a branch and a switch share a successor.
# When it fires, delete the LowerSwitchPass/StructurizeCFGPass pair at the end
# of addVulkanLinkTimePasses (llvm_passes/HipPasses.cpp, WORKAROUND #1738) and
# this test together, then close #1738 naming the LLVM change that fixed it.
set -u
LLC="@LLVM_TOOLS_BINARY_DIR@/llc"
SPIRV_VAL="@CMAKE_BINARY_DIR@/external/spirv-tools/bin/spirv-val"
[ -x "${SPIRV_VAL}" ] || SPIRV_VAL=$(command -v spirv-val || true)
[ -x "${SPIRV_VAL}" ] || { echo "FAIL: spirv-val not found"; exit 1; }
cd "$(mktemp -d)"
cat > r.ll <<'IR'
target triple = "spirv-unknown-vulkan1.3-compute"
define internal i32 @f(i1 %c, i32 %x) {
entry:
  br i1 %c, label %sw, label %other
other:
  br label %exit
sw:
  switch i32 %x, label %exit [ i32 1, label %other
                               i32 2, label %a ]
a:
  br label %exit
exit:
  %r = phi i32 [ 1, %other ], [ 2, %sw ], [ 3, %a ]
  ret i32 %r
}
define void @main() #0 {
  %r = call i32 @f(i1 true, i32 1)
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
IR
"$LLC" -mtriple=spirv-unknown-vulkan1.3-compute r.ll -filetype=obj -o r.spv > llc.log 2>&1 ||
  { echo "FAIL: llc rejected the module"; cat llc.log; exit 1; }
if "$SPIRV_VAL" --target-env vulkan1.3 r.spv > val.log 2>&1; then
  echo "CANARY FIRED: the structurizer output now validates, see the header"; exit 1
fi
grep -q "does not structurally dominate" val.log ||
  { echo "FAIL: spirv-val rejected the module for another reason"; cat val.log; exit 1; }
echo PASSED
