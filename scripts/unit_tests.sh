#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <debug|release> <llvm-15|llvm-16> ..."
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

source /opt/intel/oneapi/setvars.sh intel64 &> /dev/null
source /etc/profile.d/modules.sh
export MODULEPATH=$MODULEPATH:/home/pvelesko/modulefiles:/opt/intel/oneapi/modulefiles
export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1
export CHIP_LOGLEVEL=err

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

# Test PoCL CPU
echo "begin cpu_pocl_failed_tests"
module load opencl/pocl-cpu-$LLVM
module list
ctest --timeout 180 -j 8 --output-on-failure -E "`cat ./test_lists/cpu_pocl_failed_tests.txt`" | tee cpu_pocl_make_check_result.txt
module unload opencl/pocl-cpu-$LLVM
echo "end cpu_pocl_failed_tests"

# Test Level Zero iGPU
echo "begin igpu_level0_failed_tests"
module load levelzero/igpu
module list
ctest --timeout 180 -j 1 --output-on-failure -E "`cat ./test_lists/igpu_level0_failed_tests.txt`" | tee igpu_level0_make_check_result.txt
module unload levelzero/igpu
echo "end igpu_level0_failed_tests"

# Test Level Zero dGPU
echo "begin dgpu_level0_failed_tests"
module load levelzero/dgpu
module list
ctest --timeout 180 -j 1 --output-on-failure -E "`cat ./test_lists/dgpu_level0_failed_tests.txt`" | tee dgpu_level0_make_check_result.txt
module unload levelzero/dgpu
echo "end dgpu_level0_failed_tests"

# Test OpenCL iGPU
echo "begin igpu_opencl_failed_tests"
module load opencl/intel-igpu
module list
ctest --timeout 180 -j 8 --output-on-failure -E "`cat ./test_lists/igpu_opencl_failed_tests.txt`" | tee igpu_opencl_make_check_result.txt
module unload opencl/intel-igpu
echo "end igpu_opencl_failed_tests"

# Test OpenCL dGPU
echo "begin dgpu_opencl_failed_tests"
module load opencl/intel-dgpu
module list
ctest --timeout 180 -j 8 --output-on-failure -E "`cat ./test_lists/dgpu_opencl_failed_tests.txt`" | tee dgpu_opencl_make_check_result.txt
module unload opencl/intel-dgpu
echo "end dgpu_opencl_failed_tests"

# Test OpenCL CPU
echo "begin cpu_opencl_failed_tests"
module load opencl/intel-cpu
module list
ctest --timeout 180 -j 8 --output-on-failure -E "`cat ./test_lists/cpu_opencl_failed_tests.txt`" | tee cpu_opencl_make_check_result.txt
module unload opencl/intel-cpu
echo "end cpu_opencl_failed_tests"

function check_tests {
  file="$1"
  if grep -q "0 tests failed out of" "$file"; then
    echo "PASS"
    return 0
  else
    echo "FAIL"
    grep -E "The following tests FAILED:" -A 1000 "$file" | sed '/^$/q' | tail -n +2
    return 1
  fi
}

overall_status=0

echo "RESULTS:"
for test_result in igpu_opencl_make_check_result.txt \
                   dgpu_opencl_make_check_result.txt \
                   cpu_opencl_make_check_result.txt \
                   igpu_level0_make_check_result.txt \
                   dgpu_level0_make_check_result.txt \
                   cpu_pocl_make_check_result.txt
do
  echo -n "${test_result}: "
  check_tests "${test_result}"
  test_status=$?
  if [ $test_status -eq 1 ]; then
    overall_status=1
  fi
done

if [ $overall_status -eq 0 ]; then
  exit 0
else
  exit 1
fi
