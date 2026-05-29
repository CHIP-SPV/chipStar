#!/bin/bash
# Reproduces: chipStar does not produce a libhiprtc.so file. libCEED (and
# other HIP-7 consumers) link with `-lhiprtc` because hipconfig --version
# reports 7.x; the linker then fails with "cannot find -lhiprtc".
set -eu

BUILD_DIR=@CMAKE_BINARY_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=${BUILD_DIR}/bin/hipcc

mkdir -p ${OUT_DIR}
cd ${OUT_DIR}

cat > consumer.c <<'EOF'
int main(void) { return 0; }
EOF

${HIPCC} consumer.c -L${BUILD_DIR} -lhiprtc -o consumer 2> link.err || {
    echo "FAIL: linker could not find -lhiprtc"
    cat link.err
    exit 1
}

echo "PASS"
