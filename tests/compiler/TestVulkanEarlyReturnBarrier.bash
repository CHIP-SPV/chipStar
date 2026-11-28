#!/bin/bash
# Early returns before a barrier become a flag; other kernels stay unchanged.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestVulkanEarlyReturnBarrier.ll" -o "${OUT}"
fn() { sed -n "/^define spir_kernel void @$1(/,/^}/p" "${OUT}"; }
# Conditional branches straight into the return block.
exits() { fn "$1" | grep -c 'br i1 .*label %ret' || true; }
[ "$(exits early)" = 0 ] || { echo "FAIL: early still branches to its return"; exit 1; }
# The loop keeps only its latch exit.
[ "$(exits loop)" = 1 ] || { echo "FAIL: loop still returns early"; exit 1; }
for k in early loop; do
  fn $k | grep -q "barrier.with.group.sync" || { echo "FAIL: $k lost its barrier"; exit 1; }
done
# Both stores of early sit behind the exited flag.
[ "$(fn early | grep -c 'br i1')" -ge 2 ] || { echo "FAIL: early stores unguarded"; exit 1; }
# bound: the early return is its only exit besides the loop test.
[ "$(exits bound)" = 1 ] || { echo "FAIL: bound was not rewritten"; exit 1; }
# The work-item query stays unguarded.
fn early | grep -B1 "get_local_idj" | grep -q "br i1" && { echo "FAIL: query guarded"; exit 1; }
[ "$(exits bail)" != 0 ] || { echo "FAIL: bail was rewritten"; exit 1; }
[ "$(exits nobarrier)" != 0 ] || { echo "FAIL: nobarrier was rewritten"; exit 1; }
"@LLVM_TOOLS_BINARY_DIR@/llc" -O3 -filetype=obj "${OUT}" -o /dev/null
echo PASSED
