#!/bin/bash

# Check if at least one argument is provided
if [ $# -lt 3 ]; then
    echo "Please provide at least three arguments."
    exit 1
fi

# Check if the first argument is either "debug" or "release"
if [ "$1" != "debug" ] && [ "$1" != "release" ]; then
    echo "Error: Invalid argument. Must be either 'debug' or 'release'."
    exit 1
fi

# Set the build type based on the argument
build_type=$(echo "$1" | tr '[:lower:]' '[:upper:]')

if [ "$2" == "llvm-15" ]; then
    LLVM=llvm-15
    CLANG=clang/clang15-spirv-omp
elif [ "$2" == "llvm-16" ]; then
    LLVM=llvm-16
    CLANG=clang/clang16-spirv-omp
else
  echo "$2"
  echo "Invalid 2nd argument. Use either 'llvm-15' or 'llvm-16'."
  exit 1
fi

platform=$3
case $arg in
    "pocl-cpu"|"opencl-igpu"|"opencl-dgpu"|"opencl-cpu"|"levelzero-igpu"|"levelzero-dgpu")
        echo "The third argument is one of the specified options: $platform"
        ;;
    *)
        echo "The third argument isn't one of the specified options: $platform"
        exit 1
        ;;
esac

source /opt/intel/oneapi/setvars.sh intel64 &> /dev/null
source /etc/profile.d/modules.sh
export MODULEPATH=$MODULEPATH:/home/pvelesko/modulefiles:/opt/intel/oneapi/modulefiles
export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1

# icpx --version
# ulimit -a
sudo /opt/ocl-icd/scripts/igpu_unbind &> /dev/null
sudo /opt/ocl-icd/scripts/dgpu_unbind &> /dev/null

rm -rf HIPCC
rm -rf HIP
rm -rf bitcode/ROCm-Device-Libs
rm -rf hip-tests
rm -rf hip-testsuite

git submodule update --init
rm -rf build
rm -rf *_result.txt
mkdir build
cd build

# Use OpenCL for building/test discovery to prevent Level Zero from being used in multi-thread/multi-process environment
module load $CLANG
module load opencl/pocl-cpu-$LLVM

echo "building with $CLANG"
cmake ../ -DCMAKE_BUILD_TYPE="$build_type" &> /dev/null
make all build_tests -j &> /dev/null
echo "build complete." 
module unload opencl/pocl-cpu-$LLVM
