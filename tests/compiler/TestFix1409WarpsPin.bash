#!/bin/bash
# Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1409 and #1479.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
: > "${OUT}"
# The ballot module is separate: its __chip_ballot trips the old gate.
for IN in @TEST_NAME@ @TEST_NAME@Ballot; do
  "@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
    -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/${IN}.ll" -o - >> "${OUT}"
done
# Kernels whose lanes exchange data must be pinned to the warp size.
for K in lockstep dynshared syncwarp ballot; do
  grep -E "define spir_kernel void @$K\(.*!intel_reqd_sub_group_size" "${OUT}" || { echo "FAIL: @$K not pinned"; exit 1; }
done
# Others, and kernels reaching an indirect call, must not be.
for K in plain indirect; do
  grep -q "define spir_kernel void @$K(" "${OUT}" || { echo "FAIL: @$K missing"; exit 1; }
  if grep -E "define spir_kernel void @$K\(.*!intel_reqd_sub_group_size" "${OUT}"; then
    echo "FAIL: @$K pinned"; exit 1
  fi
done
echo PASSED
