#!/bin/bash
# Test the case where the hip/hip_fp16.h header is included before hip/hip_runtime.h 
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

export HIPCC_VERBOSE=7

${HIPCC} ${SRC_DIR}/inputs/testfp16include.cpp -o /dev/null
