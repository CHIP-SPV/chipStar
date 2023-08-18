#!/bin/bash
# Check __device__ functions and variables by the same name in multiple
# TUs are not attempted to be linked together, when -fgpu-rdc option
# is absent, and they are not mixed up at the runtime.
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}
# Also check hipcc is not issuing misplaced options.
${HIPCC} -Werror=unused-command-line-argument \
         ${SRC_DIR}/inputs/foobar{,-main}.hip -o ${OUT_DIR}/foobar
${OUT_DIR}/foobar

${HIPCC} -Werror=unused-command-line-argument \
         ${SRC_DIR}/inputs/foobar.hip -c -o ${OUT_DIR}/foobar.o
${HIPCC} -Werror=unused-command-line-argument \
         ${SRC_DIR}/inputs/foobar-main.hip -c -o ${OUT_DIR}/foobar-main.o
${HIPCC} -Werror=unused-command-line-argument \
         ${OUT_DIR}/foobar{,-main}.o -o /dev/null

${HIPCC} -Werror=unused-command-line-argument -fno-gpu-rdc \
         ${SRC_DIR}/inputs/foobar{,-main}.hip -o /dev/null
