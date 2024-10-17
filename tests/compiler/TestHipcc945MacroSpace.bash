#!/bin/bash
# Test hipcc handling of macros with spaces
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}
touch foo.c

${HIPCC} -DPACKAGE_STRING=\"H\ H\" -c foo.c

