#!/bin/bash
# A large copy out of a constant table becomes constant stores.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestVulkanConstTableCopy.ll" -o "${OUT}"
! grep -q "load i8, ptr addrspace(1)" "${OUT}" || { echo "FAIL: table copied as bytes"; exit 1; }
echo PASSED
