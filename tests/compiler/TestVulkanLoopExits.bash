#!/bin/bash
# A loop with two exits gets one exit before StructurizeCFG.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestVulkanLoopExits.ll" -o "${OUT}"
grep -q "loop.exit.guard" "${OUT}" || { echo "FAIL: loop exits not unified"; exit 1; }
echo PASSED
