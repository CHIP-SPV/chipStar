#!/bin/bash
# A load through a select of private variables must not select pointers.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestVulkanSelectLoad.ll" -o "${OUT}"
! grep -q "select i1 .*, ptr " "${OUT}" || { echo "FAIL: pointer select survived"; exit 1; }
grep -q "alloca" "${OUT}" && { echo "FAIL: private variables not promoted"; exit 1; }
"@LLVM_TOOLS_BINARY_DIR@/llc" -O3 -filetype=obj "${OUT}" -o /dev/null
echo PASSED
