#!/bin/bash
# RDC linking must not emit "Linking two modules of different data layouts".
#
# libocml_host_math_funcs.a is a host-only static library. When it is passed
# to the link line as -locml_host_math_funcs, clang's static-device-library
# handling unbundles every member into an (empty) device bitcode archive and
# llvm-link then warns about each empty module's missing data layout.
# See CHIP-SPV/chipStar#525.
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}
${HIPCC} -fgpu-rdc ${SRC_DIR}/inputs/a.hip -c -o ${OUT_DIR}/a.o
${HIPCC} -fgpu-rdc ${SRC_DIR}/inputs/b.hip -c -o ${OUT_DIR}/b.o

LINK_LOG=${OUT_DIR}/link.log
${HIPCC} -fgpu-rdc ${OUT_DIR}/{a,b}.o -o ${OUT_DIR}/ab 2>${LINK_LOG} \
  || { cat ${LINK_LOG}; exit 1; }
cat ${LINK_LOG}

if grep -q "different data layouts" ${LINK_LOG}; then
  echo "FAIL: spurious data layout warning at RDC link"
  exit 1
fi
echo PASSED
