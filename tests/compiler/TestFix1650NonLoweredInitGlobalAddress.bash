#!/bin/bash
# Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1650.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/@TEST_NAME@.ll" -o "${OUT}"
if grep -n poison "${OUT}"; then echo FAIL; exit 1; fi
for G in _ZZ4getPvE1P __const._Z1kPi.p; do
  grep -q "store ptr addrspace(4) %.*, ptr addrspace(1) @${G}," "${OUT}" ||
    { echo "FAIL: nothing initializes @${G}"; exit 1; }
done
grep -q "^@__const._Z1kPi.p = .* global " "${OUT}" ||
  { echo "FAIL: @__const._Z1kPi.p is written at runtime but still constant"; exit 1; }
echo PASSED
