#!/bin/bash
source /etc/profile.d/modules.sh &> /dev/null
HIP_DIR=$1
echo "Building LibCEED with HIP_DIR: ${HIP_DIR}"
if which clang >/dev/null 2>&1; then
    echo "clang is in the PATH"
else
    echo "Error: clang is not in the PATH"
    exit 1
fi

if which clang++ >/dev/null 2>&1; then
    echo "clang++ is in the PATH"
else
    echo "Error: clang++ is not in the PATH"
    exit 1
fi

module load HIP/hipBLAS mkl
rm -rf libCEED
git clone https://github.com/CHIP-SPV/libCEED.git -b jed/chip-spv-pvelesko
cd libCEED
make FC= CC=clang CXX=clang++ BACKENDS="/gpu/hip/ref /gpu/hip/shared /gpu/hip/gen" -j