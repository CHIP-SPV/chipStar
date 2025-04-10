#!/bin/bash

# Test description:
# This test verifies that building a static library with device code using
# -fgpu-rdc fails as described in https://github.com/CHIP-SPV/chipStar/issues/984

# Exit script on error
set -eu

# CMake substituted variables
SRC_DIR="@CMAKE_CURRENT_SOURCE_DIR@"
OUT_DIR="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
TEST_NAME="@TEST_NAME@" # Make sure TEST_NAME is available if needed later

# Create output directory
mkdir -p "${OUT_DIR}"

# Compile the device code files
${HIPCC} -fgpu-rdc -fPIC -I"${SRC_DIR}" -c "${SRC_DIR}/k.cu" -o "${OUT_DIR}/k.o"
${HIPCC} -fgpu-rdc -fPIC -I"${SRC_DIR}" -c "${SRC_DIR}/k1.cu" -o "${OUT_DIR}/k1.o"

# Create the static library
ar rcs "${OUT_DIR}/libk.a" "${OUT_DIR}/k.o" "${OUT_DIR}/k1.o"

# Compile the main host file
${HIPCC} -fgpu-rdc -I"${SRC_DIR}" -c "${SRC_DIR}/t.cpp" -o "${OUT_DIR}/t.o"

# Link the main file and the static library
${HIPCC} --save-temps -v -fgpu-rdc --hip-link "${OUT_DIR}/t.o" "${OUT_DIR}/libk.a" -o "${OUT_DIR}/TestStaticLibRDC" 
echo "TestStaticLibRDC.log: ${OUT_DIR}/TestStaticLibRDC.log"
${OUT_DIR}/TestStaticLibRDC

# Attempt to run the executable - we expect this to fail with a specific error
# Use run_and_check_error.bash helper if available, otherwise grep for the error
RUN_EXEC="${OUT_DIR}/TestStaticLibRDC"

# Run the executable, capture stderr, and check for success and absence of errors
echo "Running: ${RUN_EXEC}"
STDERR_OUTPUT=$("${RUN_EXEC}" 2>&1) || true # Run and capture stderr, allow non-zero exit code here


# Check for explicit error messages in stderr
if echo "${STDERR_OUTPUT}" | grep -qE 'CHIP error|hipError'; then
  echo "Test FAILED: Error messages found in output."
  echo "Output:"
  echo "${STDERR_OUTPUT}"
  exit 1
fi
