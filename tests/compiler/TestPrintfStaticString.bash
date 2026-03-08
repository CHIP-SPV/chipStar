#!/bin/bash
# Test that printf with static strings produces the correct runtime output from multiple threads.
set -eu

SRC_DIR=@CMAKE_CURRENT_SOURCE_DIR@
OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

SOURCE_FILE=${SRC_DIR}/TestPrintfStaticString.hip
EXECUTABLE_NAME=${OUT_DIR}/TestPrintfStaticString.out
EXPECTED_LINE="File main.hip errored"
EXPECTED_COUNT=32

mkdir -p ${OUT_DIR}

# Compile the HIP source file
echo "Compiling ${SOURCE_FILE}..."
${HIPCC} ${SOURCE_FILE} -o ${EXECUTABLE_NAME}

# Check if compilation was successful
if [ ! -f "${EXECUTABLE_NAME}" ]; then
    echo "Compilation FAILED"
    exit 1
fi
echo "Compilation successful: ${EXECUTABLE_NAME} created."

# Run the executable and capture its output
echo "Running ${EXECUTABLE_NAME}..."
ACTUAL_OUTPUT=$(${EXECUTABLE_NAME})

# Count the occurrences of the expected line in the output
ACTUAL_COUNT=$(echo "${ACTUAL_OUTPUT}" | grep -Fx "${EXPECTED_LINE}" -c || true)
# The '|| true' prevents the script from exiting if grep finds 0 matches (which returns exit code 1)

# Check the output count
echo "-------------------------------------------"
echo "Expected line:     '${EXPECTED_LINE}'"
echo "Expected count:    ${EXPECTED_COUNT}"
echo "Actual output:"
echo "${ACTUAL_OUTPUT}"
echo "-------------------------------------------"

if [ "${ACTUAL_COUNT}" -eq "${EXPECTED_COUNT}" ]; then
    echo "SUCCESS: Runtime output contains the expected number of matching lines."
    # Optional: Clean up the output directory on success
    # rm -rf ${OUT_DIR}
    exit 0
else
    echo "FAILURE: Runtime output does not contain the expected number of matching lines."
    # Keep OUT_DIR for inspection on failure
    exit 1
fi
