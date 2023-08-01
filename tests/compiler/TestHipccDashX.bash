#!/bin/bash
# Test hipcc handling of -x hip
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}
touch source1.xyz
touch source2.foo

${HIPCC} -x hip ./source1.xyz ./source2.foo -c

