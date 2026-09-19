#!/bin/bash
# Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1640.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestFix1640AggregateInitGlobalAddress.ll" -o "${OUT}"
if grep -E "store .*poison" "${OUT}"; then echo FAIL; exit 1; fi
echo PASSED
