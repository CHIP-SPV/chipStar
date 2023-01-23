#!/bin/bash
export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1

source /opt/intel/oneapi/setvars.sh intel64

git submodule update --init
mkdir build
cd build

# TODO check Release as well 
cmake ../ -DLLVM_CONFIG=/usr/bin/llvm-config-15
make -j
make build_tests -j

# Test OpenCL iGPU
echo "begin igpu_opencl_failed_tests"
CHIP_BE=opencl CHIP_DEVICE_TYPE=gpu CHIP_PLATFORM=4 CHIP_DEVICE=0 ctest --timeout 180 -j 1 --output-on-failure -E "`cat ./test_lists/igpu_opencl_failed_tests.txt`" | tee igpu_opencl_make_check_result.txt
echo "end igpu_opencl_failed_tests"
# Test OpenCL dGPU
echo "begin dgpu_opencl_failed_tests"
CHIP_BE=opencl CHIP_DEVICE_TYPE=gpu CHIP_PLATFORM=3 CHIP_DEVICE=0 ctest --timeout 180 -j 1 --output-on-failure -E "`cat ./test_lists/dgpu_opencl_failed_tests.txt`" | tee dgpu_opencl_make_check_result.txt
echo "end dgpu_opencl_failed_tests"
# Test OpenCL CPU
echo "begin cpu_opencl_failed_tests"
CHIP_BE=opencl CHIP_DEVICE_TYPE=cpu CHIP_PLATFORM=1 CHIP_DEVICE=0 ctest --timeout 180 -j 1 --output-on-failure -E "`cat ./test_lists/cpu_opencl_failed_tests.txt`" | tee cpu_opencl_make_check_result.txt
echo "end cpu_opencl_failed_tests"
# Test Level Zero iGPU
echo "begin igpu_level0_failed_tests"
CHIP_BE=level0 CHIP_DEVICE=1 ctest --timeout 180 -j 1 --output-on-failure -E "`cat ./test_lists/igpu_level0_failed_tests.txt`" | tee igpu_level0_make_check_result.txt
echo "end igpu_level0_failed_tests"
# Test Level Zero dGPU
echo "begin dgpu_level0_failed_tests"
CHIP_BE=level0 CHIP_DEVICE=0 ctest --timeout 180 -j 1 --output-on-failure -E "`cat ./test_lists/dgpu_level0_failed_tests.txt`" | tee dgpu_level0_make_check_result.txt
echo "end dgpu_level0_failed_tests"


if [[ `cat igpu_opencl_make_check_result.txt | grep "0 tests failed out of"` ]]
then
    echo "iGPU OpenCL PASS"
    iGPU_OpenCL=1
else
    echo "iGPU OpenCL FAIL"
    iGPU_OpenCL=0
fi

if [[ `cat dgpu_opencl_make_check_result.txt | grep "0 tests failed out of"` ]]
then
    echo "dGPU OpenCL PASS"
    dGPU_OpenCL=1
else
    echo "dGPU OpenCL FAIL"
    dGPU_OpenCL=0
fi

if [[ `cat cpu_opencl_make_check_result.txt | grep "0 tests failed out of"` ]]
then
    echo "CPU OpenCL PASS"
    CPU_OpenCL=1
else
    echo "CPU OpenCL FAIL"
    CPU_OpenCL=0
fi

if [[ `cat igpu_level0_make_check_result.txt | grep "0 tests failed out of"` ]]
then
    echo "iGPU OpenCL PASS"
    iGPU_LevelZero=1
else
    echo "iGPU OpenCL FAIL"
    iGPU_LevelZero=0
fi

if [[ `cat dgpu_level0_make_check_result.txt | grep "0 tests failed out of"` ]]
then
    echo "dGPU Level zero PASS"
    dGPU_LevelZero=1
else
    echo "dGPU Level Zero FAIL"
    dGPU_LevelZero=0
fi

if [[ "$iGPU_OpenCL" -eq "1" && \
      "$dGPU_OpenCL" -eq "1" && \
      "$CPU_OpenCL" -eq "1" && \
      "$iGPU_LevelZero" -eq "1" && \
      "$dGPU_LevelZero" -eq "1" ]]
then
    echo "ALL PASS"
    exit 0
else
    echo "ALL FAIL"
    exit 1
fi