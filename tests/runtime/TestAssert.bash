#!/bin/bash
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}
cd ${OUT_DIR}

${HIPCC} ${SRC_DIR}/inputs/Assert.hip -o assert

! ./assert >output.log 2>&1 || {
    echo "FAIL: Expected error code."
    exit 1
}

grep -c "Assert.hip:5: void k(): Device-side assertion .false [&][&] \"Hello, World!\". failed." \
     output.log ||  {
    echo "FAIL: expected assertion error message was not found."
    exit 1
}
