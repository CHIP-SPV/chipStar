#!/bin/bash
# Regression test for CHIP-SPV/chipStar#1680: no phi in the lowered device IR may list a predecessor twice.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"
rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"
"@CMAKE_BINARY_DIR@/bin/hipcc" -O3 -fno-jump-tables --save-temps=cwd -c "@CMAKE_CURRENT_SOURCE_DIR@/TestSpirvDuplicatePhiHip.hip" -o t.o
"@LLVM_TOOLS_BINARY_DIR@/llvm-dis" ./*-lower.bc -o lower.ll
awk -F '[][]' '/ = phi i32 / { n++; delete seen; for (i = 2; i < NF; i += 2) { sub(/.*, */, "", $i); if (seen[$i]++) bad = 1 } } END { exit bad || !n }' lower.ll
echo PASSED
