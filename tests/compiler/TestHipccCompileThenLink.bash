#!/bin/bash
# Test the case where object files are compiled separately and then linked 
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

export HIPCC_VERBOSE=7
mkdir -p ${OUT_DIR}
echo "Step #1"
${HIPCC} -c ${SRC_DIR}/TestHipccCompileThenLinkKernel.cpp -o ${OUT_DIR}/TestHipccCompileThenLinkKernel.cc.o
echo "Step #2"
${HIPCC} -c ${SRC_DIR}/TestHipccCompileThenLinkMain.cpp -o ${OUT_DIR}/TestHipccCompileThenLinkMain.cc.o
echo "Step #3"
${HIPCC}  ${OUT_DIR}/TestHipccCompileThenLinkKernel.cc.o  ${OUT_DIR}/TestHipccCompileThenLinkMain.cc.o  -o TestHipccCompileThenLink 
