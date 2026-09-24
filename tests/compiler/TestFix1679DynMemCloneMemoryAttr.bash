#!/bin/bash
# Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1679.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestFix1679DynMemCloneMemoryAttr.ll" -o "${OUT}"
grep -q "smem__hidden_dyn_local_mem" "${OUT}" || { echo "FAIL: h was not lowered"; exit 1; }
if grep "argmem: none" "${OUT}"; then echo FAIL; exit 1; fi
echo PASSED
