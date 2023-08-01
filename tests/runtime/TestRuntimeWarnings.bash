# A regression test for https://github.com/CHIP-SPV/chipStar/issues/481.
#!/bin/bash
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}
cd ${OUT_DIR}

${HIPCC} ${SRC_DIR}/inputs/MissingDef.hip -o missing-def

# Executable may fail or crash - that's expected.
CHIP_LOGLEVEL=warn ./missing-def >output.log 2>&1 || true

# Should not display warning on symbols wich are decorated with
# 'BuiltIn' attribute.
! grep -c "warning .* Missing definition for '__spirv_BuiltIn" output.log ||  {
    echo "FAIL: unexpected warning was seen."
    exit 1
}

grep -c "warning .* Missing definition for '_Z3fooi'" output.log ||  {
    echo "FAIL: expected warning was seen."
    exit 1
}
