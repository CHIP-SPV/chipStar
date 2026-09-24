#!/bin/bash
# Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1678.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/@TEST_NAME@.ll" -o "${OUT}"
# The wrapper kernel must call the original kernel as spir_func, passing byval.
grep -E "call spir_func void @k\.original_kernel\(.*byval\(%struct\.Big\) align 64" "${OUT}" || { echo FAIL; exit 1; }
# The copy it passes must be as aligned as the byval parameter says.
grep -E "alloca %struct\.Big, align 64" "${OUT}" || { echo FAIL; exit 1; }
echo PASSED
