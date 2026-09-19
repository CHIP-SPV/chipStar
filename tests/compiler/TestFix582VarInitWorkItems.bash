#!/bin/bash
# Reproducer for CHIP-SPV/chipStar#582: program-scope variable init kernels
# must spread their work across the work items of the launch.
set -eu

HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_DIS="@CLANG_ROOT_PATH_BIN@/llvm-dis"
SRC="@CMAKE_SOURCE_DIR@/tests/runtime/TestGlobalVarInit.hip"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"
mkdir -p "${OUT}"
cd "${OUT}"
"${HIPCC}" --save-temps=cwd -c "${SRC}" -o TestGlobalVarInit.o
"${LLVM_DIS}" "$(ls ./*-lower.bc | head -1)" -o lowered.ll

if ! awk '/^define .*@__chip_var_init_/ { f = 1 }
          f && /call .*@_Z15get_global_sizej/ { found = 1 }
          /^}/ { f = 0 }
          END { exit !found }' lowered.ll; then
  echo "FAIL: no variable init kernel calls get_global_size"
  exit 1
fi
echo "PASSED"
