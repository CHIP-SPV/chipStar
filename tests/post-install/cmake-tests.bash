#!/bin/bash
set -eu

CXX=@CMAKE_CXX_COMPILER@

# Changes directory to an empty directory.
function setup-test-dir {
    cd @CMAKE_CURRENT_BINARY_DIR@
    rm -rf test-build
    mkdir test-build
    cd test-build
}

echo "### Check hip::device CMake target"
setup-test-dir
cmake @CMAKE_CURRENT_SOURCE_DIR@/HipDeviceTarget \
      -DCMAKE_PREFIX_PATH=@CMAKE_INSTALL_PREFIX@ \
      -DCMAKE_CXX_COMPILER=${CXX}
make  VERBOSE=1
echo "Success"
echo
echo "### Check hip::host Cmake target"
if [[ "$(basename ${CXX})" =~ clang++.* || "$(basename ${CXX})" =~ g++.* ]]
then
    setup-test-dir
    cmake @CMAKE_CURRENT_SOURCE_DIR@/HipHostTarget \
	  -DCMAKE_PREFIX_PATH=@CMAKE_INSTALL_PREFIX@ \
	  -DCMAKE_CXX_COMPILER=${CXX}
    make VERBOSE=1
    echo "Success"
else
    echo "Skip: clang++ or g++ is required"
fi
