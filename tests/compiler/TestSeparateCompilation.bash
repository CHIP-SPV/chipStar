#!/bin/bash
# Test separate compilation with -dc flag (issue #893).
# The -dc flag is equivalent to -fgpu-rdc -c and is used in separate
# compilation workflows. When linking objects compiled with -dc, the
# linker must perform device code linking to generate __hip_fatbin.
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

mkdir -p ${OUT_DIR}

# Compile each file with -dc (device compilation for separate compilation)
${HIPCC} -dc ${SRC_DIR}/inputs/a.hip -o ${OUT_DIR}/a.o
${HIPCC} -dc ${SRC_DIR}/inputs/b.hip -o ${OUT_DIR}/b.o

# Link - must pass -fgpu-rdc so hipcc knows to add --hip-link
${HIPCC} -fgpu-rdc ${OUT_DIR}/{a,b}.o -o ${OUT_DIR}/ab

# Run
${OUT_DIR}/ab
