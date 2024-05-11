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

rm -rf libCEED
git clone https://github.com/CHIP-SPV/libCEED.git -b chipStar
cd libCEED
make FC= CC=hipcc CXX=hipcc BACKENDS="/gpu/hip/ref /gpu/hip/shared /gpu/hip/gen" -j 