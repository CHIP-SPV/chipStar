#!/bin/bash
# A byte stride GEP read as floats steps in a float array.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestVulkanByteStrideGEP.ll" -o "${OUT}"
! grep -q "getelementptr inbounds \[24 x i8\]" "${OUT}" || { echo "FAIL: byte stride kept"; exit 1; }
"@LLVM_TOOLS_BINARY_DIR@/llc" -O3 -filetype=obj "${OUT}" -o /dev/null
echo PASSED
