#!/bin/bash

# Set the path to hipcc - it should be in the build directory
HIPCC_PATH=/space/pvelesko/chipStar/main/build/bin/hipcc

# Check if hipcc exists
if [ ! -f "$HIPCC_PATH" ]; then
    echo "Error: hipcc not found at $HIPCC_PATH"
    exit 1
fi

# Expected order of Linker arguments
EXPECTED_ARGS="-L/path/to/libA -L/path/to/libB -llibA -llibB"
# Create a grep pattern that respects the order
# Replace spaces with .* to allow for other arguments inserted by hipcc/linker
GREP_PATTERN=$(echo "$EXPECTED_ARGS" | sed 's/ /.*/g')

# Minimal C++ source code
SOURCE_CODE="int main() { return 0; }"

# Run hipcc with verbose output, specific argument order, reading source from stdin
# NO -c flag, so linking will happen (and likely fail, but we check args before that)
OUTPUT=$(echo "$SOURCE_CODE" | "$HIPCC_PATH" -v -x c++ - -o /dev/null $EXPECTED_ARGS 2>&1)

# Debug: Print the grep pattern and the output line being checked
echo "Debug: Grep pattern: $GREP_PATTERN"
LINKER_CMD_LINE=$(echo "$OUTPUT" | grep "/ld" | head -n 1) # Get the first line containing /ld
echo "Debug: Checking linker command line:"
echo "$LINKER_CMD_LINE"

echo "Debug: Starting grep check..."
# Check if the verbose output contains the ld command with the expected argument order
# Use grep -E for extended regex and check the specific line invoking the linker
# Use -e to specify the pattern explicitly
if echo "$LINKER_CMD_LINE" | grep -q -E -e "$GREP_PATTERN"; then
    echo "Test Passed: Linker argument order appears to be preserved."
    exit 0
else
    echo "Test Failed: Linker argument order might be changed."
    echo "--- hipcc -v output ---"
    echo "$OUTPUT"
    echo "-----------------------"
    exit 1
fi 