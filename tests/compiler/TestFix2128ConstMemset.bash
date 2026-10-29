#!/bin/bash
# Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2128.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestFix2128ConstMemset.ll" -o "${OUT}"
grep -q "call void @llvm.memset.p1.i64(.*i64 64," "${OUT}" || { echo "FAIL: constant memset was expanded"; exit 1; }
[ "$(grep -c "call void @llvm.memset" "${OUT}")" = 1 ] || { echo "FAIL: a memset that needs expanding survived"; exit 1; }
echo PASSED
