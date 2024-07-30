#!/bin/bash

set -e

HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

# Create dummy input files
touch main.o meshBasis.o
mkdir -p BlasLapack
touch BlasLapack/libBlasLapack.a

# Run hipcc with the specified order
$HIPCC ./main.o ./meshBasis.o ./BlasLapack/libBlasLapack.a -v 2>&1 | tee hipcc_output.log

# Check if the order is preserved in the output
if grep -q "main.o.*meshBasis.o.*BlasLapack/libBlasLapack.a" hipcc_output.log; then
    echo "Test passed: File order is preserved"
    exit 0
else
    echo "Test failed: File order is not preserved"
    exit 1
fi

# Clean up
rm -f main.o meshBasis.o hipcc_output.log
rm -rf BlasLapack