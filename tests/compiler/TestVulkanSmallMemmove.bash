#!/bin/bash
# A small constant memmove must not compare pointers.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestVulkanSmallMemmove.ll" -o "${OUT}"
! grep -q "icmp .* ptr " "${OUT}" || { echo "FAIL: memmove compares pointers"; exit 1; }
"@LLVM_TOOLS_BINARY_DIR@/llc" -O3 -filetype=obj "${OUT}" -o /dev/null
echo PASSED
