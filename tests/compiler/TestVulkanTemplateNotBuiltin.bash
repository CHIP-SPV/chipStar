#!/bin/bash
# A template instance named like an OpenCL builtin is not lowered as one.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestVulkanTemplateNotBuiltin.ll" -o "${OUT}"
! grep -q "llvm.fma" "${OUT}" || { echo "FAIL: template lowered as fma"; exit 1; }
grep -q "store i32 5" "${OUT}" || { echo "FAIL: template body lost"; exit 1; }
echo PASSED
