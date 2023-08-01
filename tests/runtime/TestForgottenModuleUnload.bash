# Regression test for https://github.com/CHIP-SPV/chipStar/issues/373.
#!/bin/bash
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}
cd ${OUT_DIR}

${HIPCC} ${SRC_DIR}/inputs/ForgetModuleUnload.hip -o prog

CHIP_LOGLEVEL=warn ./prog ${SRC_DIR}/inputs/Module.bin >output.log 2>&1 || {
    echo "FAIL: Expected error code 0."
    exit 1
}

PATTERN="CHIP warning .* Program still has unloaded HIP modules at program exit"
grep -c "$PATTERN" output.log || {
    echo "FAIL: expected pattern was not found:"
    echo "Pattern: '$PATTERN'"
    exit 1
}
