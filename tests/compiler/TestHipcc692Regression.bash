#!/bin/bash
# Check that -MD -MT options are respected
set -eu

BIN_DIR=@CMAKE_BINARY_DIR@
SRC_DIR=@CMAKE_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}

touch ${OUT_DIR}/test_hipcub_basic.cpp
mkdir -p ${OUT_DIR}/CMakeFiles/test_hipcub_basic.dir

export HIPCC_VERBOSE=7

${HIPCC} \
-DGTEST_LINKED_AS_SHARED_LIBRARY=1  \
-O3 -DNDEBUG  \
-x hip -D__HIP_PLATFORM_SPIRV__= --offload=spirv64 -nohipwrapperinc --hip-path=${BIN_DIR} --target=x86_64-unknown-linux-gnu  \
-include ${SRC_DIR}/include/hip/spirv_fixups.h -std=c++14  \
-c ${OUT_DIR}/test_hipcub_basic.cpp \
-MD -MT ${OUT_DIR}/CMakeFiles/test_hipcub_basic.dir/test_hipcub_basic.cpp.o  \
-MF ${OUT_DIR}/CMakeFiles/test_hipcub_basic.dir/test_hipcub_basic.cpp.o.d \
-o ${OUT_DIR}/CMakeFiles/test_hipcub_basic.dir/test_hipcub_basic.cpp.o \
| tee TestHipcc692ReggressionOutput.log
# make sure that -MT <file> is respected
grep -- "-MT ${OUT_DIR}/CMakeFiles/test_hipcub_basic.dir/test_hipcub_basic.cpp.o" ./TestHipcc692ReggressionOutput.log
if [ $? -ne 0 ]; then
    echo "-MT not respected"
    exit 1
fi

grep -- "-MF ${OUT_DIR}/CMakeFiles/test_hipcub_basic.dir/test_hipcub_basic.cpp.o.d" ./TestHipcc692ReggressionOutput.log 
if [ $? -ne 0 ]; then
    echo "-MF not respected"
    exit 1
fi
