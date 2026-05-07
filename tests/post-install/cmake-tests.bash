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
echo
echo "### Check hip::device with mixed C/C++ sources (#1001)"
setup-test-dir
cmake @CMAKE_CURRENT_SOURCE_DIR@/HipDeviceCSource \
      -DCMAKE_PREFIX_PATH=@CMAKE_INSTALL_PREFIX@ \
      -DCMAKE_CXX_COMPILER=${CXX}
make VERBOSE=1
echo "Success"
echo

echo "### Check legacy FindHIP cmake integration on SPIR-V platform (run_hipcc.cmake spirv branch)"
# Regression test: when HIP_PLATFORM=spirv, run_hipcc.cmake must use the clang
# branch, not the nvcc branch. The nvcc branch inserts '--compiler-options -fPIC'
# which clang does not understand, causing compilation to fail.
setup-test-dir
cmake @CMAKE_CURRENT_SOURCE_DIR@/FindHIPLegacySpirvTarget \
      -DCMAKE_PREFIX_PATH=@CMAKE_INSTALL_PREFIX@ \
      -DCMAKE_MODULE_PATH=@CMAKE_INSTALL_PREFIX@/lib/cmake/hip \
      -DCMAKE_CXX_COMPILER=${CXX}
# Build only the compilation step (linking requires hipcc_cmake_linker_helper, tested separately).
# run_hipcc.cmake names the output file foo_generated_foo.hip.o
ninja -v "CMakeFiles/foo.dir/foo_generated_foo.hip.o" 2>&1 | tee build.log
if grep -qF "''-fPIC''" build.log; then
    echo "FAIL: NVCC-style quoted -fPIC found — run_hipcc.cmake spirv branch missing"
    exit 1
fi
if [ ! -f CMakeFiles/foo.dir/foo_generated_foo.hip.o ]; then
    echo "FAIL: foo_generated_foo.hip.o was not produced — compilation failed"
echo "### check CMake-CUDA"
setup-test-dir

export CUDACXX="@CMAKE_INSTALL_PREFIX@"/bin/cucc

# Magic string for CMake to enable CUDA compilation on chipStar.
export CUCC_VERSION_STRING="nvcc: NVIDIA (R) Cuda compiler driver"

# CMake needs to find libcudart.so, libcudart_static.a and
# libcudadevrt.so to succeed. An alternate way is to place these them
# under @CMAKE_INSTALL_PREFIX@"/lib.
export LIBRARY_PATH=@CMAKE_CURRENT_SOURCE_DIR@/CMake-CUDA:"${LIBRARY_PATH-}"

cmake "@CMAKE_CURRENT_SOURCE_DIR@"/CMake-CUDA \
      -DCMAKE_CUDA_ARCHITECTURES="72" \
      -DCMAKE_BUILD_RPATH="@CMAKE_INSTALL_PREFIX@"/lib > configure.log 2>&1

# Check the CMake didn't pick up the real nvcc and CUDA toolkit.
grep -q "CMAKE_CUDA_COMPILER_TOOLKIT_ROOT: @CMAKE_INSTALL_PREFIX@" configure.log
grep -q "CMAKE_CUDA_COMPILER: .*cucc" configure.log

# Check we are using CMake's NVIDIA compilation toolchain.
grep -q "CMAKE_CUDA_COMPILER_ID: NVIDIA" configure.log

# Try to build and run a CUDA executable.
make -j1
CHIP_LOGLEVEL=warn ./myexe | grep -q 'Hello, World!'

unset CUDACXX CUCC_VERSION_STRING
echo "Success"
