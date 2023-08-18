#!/bin/bash
# Test RDC mode with single compile command.
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

export HIPCC_VERBOSE=7
mkdir -p ${OUT_DIR}
# Also check hipcc is not issuing misplaced options.
${HIPCC} ${SRC_DIR}/inputs/helloWorld.c -o ${OUT_DIR}/helloWorld.out
${OUT_DIR}/helloWorld.out
