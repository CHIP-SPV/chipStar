#!/bin/bash

# TestStaticLibRDC.bash with the explicit --offload-arch that Kokkos passes (#1408).

# Exit script on error
set -eu

# CMake substituted variables
SRC_DIR="@CMAKE_CURRENT_SOURCE_DIR@"
OUT_DIR="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"

ARCH_FLAG="--offload-arch=gfx1030"

# Create output directory
mkdir -p "${OUT_DIR}"

# Compile the device code files
${HIPCC} ${ARCH_FLAG} -fgpu-rdc -fPIC -I"${SRC_DIR}" -c "${SRC_DIR}/k.cu" -o "${OUT_DIR}/k.o"
${HIPCC} ${ARCH_FLAG} -fgpu-rdc -fPIC -I"${SRC_DIR}" -c "${SRC_DIR}/k1.cu" -o "${OUT_DIR}/k1.o"

# Create the static library
ar rcs "${OUT_DIR}/libk.a" "${OUT_DIR}/k.o" "${OUT_DIR}/k1.o"

# Compile the main host file
${HIPCC} ${ARCH_FLAG} -fgpu-rdc -I"${SRC_DIR}" -c "${SRC_DIR}/t.cpp" -o "${OUT_DIR}/t.o"

# Link the main file and the static library
${HIPCC} ${ARCH_FLAG} -fgpu-rdc "${OUT_DIR}/t.o" "${OUT_DIR}/libk.a" \
         -o "${OUT_DIR}/TestStaticLibRDCOffloadArch"

RUN_EXEC="${OUT_DIR}/TestStaticLibRDCOffloadArch"
echo "Running: ${RUN_EXEC}"
STDERR_OUTPUT=$("${RUN_EXEC}" 2>&1) || { echo "${STDERR_OUTPUT}"; exit 1; }

# The kernels live only in libk.a.
if echo "${STDERR_OUTPUT}" | grep -qE 'CHIP error|hipError'; then
  echo "Test FAILED: Error messages found in output."
  echo "Output:"
  echo "${STDERR_OUTPUT}"
  exit 1
fi
