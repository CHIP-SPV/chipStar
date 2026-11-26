#!/bin/bash
# Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2391.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestFix2391VulkanZeroMemset.ll" -o "${OUT}"
grep -q "store { \[3 x double\] } zeroinitializer, ptr addrspace(3) @lsums" "${OUT}" || { echo "FAIL: memset was not a typed zero store"; exit 1; }
echo PASSED
