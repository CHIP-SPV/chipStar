#!/bin/bash
# Regression test for hipModuleLoad (file-based loader).
# See inputs/TestModuleLoadFromFile.hip.
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}
cd ${OUT_DIR}

${HIPCC} ${SRC_DIR}/inputs/TestModuleLoadFromFile.hip -o prog

CHIP_LOGLEVEL=warn ./prog ${SRC_DIR}/inputs/Module.bin
