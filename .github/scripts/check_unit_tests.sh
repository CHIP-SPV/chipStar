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
CHIP_PLATFORM=4 CHIP_DEVICE_TYPE=gpu CHIP_BE=opencl make check | tee igpu_opencl_make_check_result.txt
# Test OpenCL dGPU
CHIP_PLATFORM=3 CHIP_DEVICE_TYPE=gpu CHIP_BE=opencl make check | tee dgpu_opencl_make_check_result.txt
# Test OpenCL CPU
CHIP_PLATFORM=1 CHIP_DEVICE_TYPE=cpu CHIP_BE=opencl make check | tee cpu_opencl_make_check_result.txt
# Test Level Zero iGPU
CHIP_DEVICE_TYPE=gpu CHIP_DEVICE=1 CHIP_BE=level0 make check | tee igpu_level0_make_check_result.txt
# Test Level Zero dGPU
CHIP_DEVICE_TYPE=gpu CHIP_DEVICE=0 CHIP_BE=level0 make check | tee dgpu_level0_make_check_result.txt


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