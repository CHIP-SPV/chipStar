#!/bin/bash
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}
cd ${OUT_DIR}

${HIPCC} ${SRC_DIR}/inputs/AssertFail.hip -o assert-fail

! ./assert-fail >output.log 2>&1 || {
    echo "FAIL: Expected error code."
    exit 1
}

grep -c "file:123: function: Device-side assertion .expression. failed." \
     output.log ||  {
    echo "FAIL: Expected assertion error message was not found."
    exit 1
}
