#!/bin/bash

# Wrapper script for TestEnvVars that unsets conflicting environment variables
# This ensures the test runs with a clean environment to avoid conflicts between
# device type filtering (CHIP_DEVICE_TYPE) and manual device selection (CHIP_PLATFORM/CHIP_DEVICE)

echo "Running TestEnvVars with clean environment..."

# Run the TestEnvVars test with completely clean environment
# This avoids conflicts from modules that set CHIP environment variables
# Check if we're already in build directory, otherwise cd to it
if [ ! -f "DartConfiguration.tcl" ]; then
    cd build
fi

# Use env -i to start with clean environment, keeping only essential variables
env -i \
    PATH="$PATH" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    HOME="$HOME" \
    USER="$USER" \
    ctest -R TestEnvVars -V

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "TestEnvVars PASSED"
else
    echo "TestEnvVars FAILED"
fi

exit $exit_code
