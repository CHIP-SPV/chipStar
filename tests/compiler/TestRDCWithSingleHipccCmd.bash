#!/bin/bash
# Test RDC mode with single compile command.
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}
# Also check hipcc is not issuing misplaced options.
${HIPCC} -Werror=unused-command-line-argument -fgpu-rdc \
         ${SRC_DIR}/inputs/{a,b}.hip -o ${OUT_DIR}/ab
${OUT_DIR}/ab
