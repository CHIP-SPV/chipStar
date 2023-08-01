#!/bin/bash
# Test the case where there's a mix of object files & source files
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

export HIPCC_VERBOSE=7
mkdir -p ${OUT_DIR}
echo "Step #1"
${HIPCC} -c ${SRC_DIR}/TestHipccCompileThenLinkKernel.cpp -o TestHipccCompileThenLinkKernel.cc.o
echo "Step #2"
${HIPCC}  TestHipccCompileThenLinkKernel.cc.o ${SRC_DIR}/TestHipccCompileThenLinkMain.cpp -o TestHipccCompileAndLink
