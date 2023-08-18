#!/bin/bash
# Test RDC mode with separate compile and link steps.
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}
# Also check hipcc is not issuing misplaced options.
${HIPCC} -Werror=unused-command-line-argument -fgpu-rdc \
         ${SRC_DIR}/inputs/a.hip -c -o ${OUT_DIR}/a.o
${HIPCC} -Werror=unused-command-line-argument -fgpu-rdc \
         ${SRC_DIR}/inputs/b.hip -c -o ${OUT_DIR}/b.o
${HIPCC} -Werror=unused-command-line-argument \
         -fgpu-rdc ${OUT_DIR}/{a,b}.o -o ${OUT_DIR}/ab
${OUT_DIR}/ab
