#!/bin/bash
# A load or store through a phi of pointers moves into the predecessors.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestVulkanPhiAccess.ll" -o "${OUT}"
! grep -q "phi ptr" "${OUT}" || { echo "FAIL: pointer phi survived"; exit 1; }
"@LLVM_TOOLS_BINARY_DIR@/llc" -O3 -filetype=obj "${OUT}" -o /dev/null
echo PASSED
