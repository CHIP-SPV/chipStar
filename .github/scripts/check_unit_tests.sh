#!/bin/bash
source /opt/intel/oneapi/setvars.sh intel64
source /etc/profile.d/modules.sh
export MODULEPATH=$MODULEPATH:/home/pvelesko/modulefiles:/opt/intel/oneapi/modulefiles
export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1

ulimit -a

git submodule update --init
mkdir build
cd build

# TODO check Release as well 
# Use OpenCL for building/test discovery to prevent Level Zero from being used in multi-thread/multi-process environment
cmake ../ -DLLVM_CONFIG=/usr/bin/llvm-config-15

module load mkl
# Load ocl-icd and intel-gpu
module load opencl/intel-gpu

# Ensure that only igpu is active for build/test discovery and OpenCL is used
sudo /opt/ocl-icd/scripts/dgpu_unbind 
sudo /opt/ocl-icd/scripts/igpu_unbind 
sudo /opt/ocl-icd/scripts/igpu_bind 
export CHIP_BE=opencl

# Build 
make -j
make build_tests -j
sudo /opt/ocl-icd/scripts/igpu_unbind &> /dev/null

# Test PoCL CPU
echo "begin cpu_pocl_failed_tests"
module swap opencl opencl/pocl-cpu
clinfo -l
CHIP_BE=opencl CHIP_DEVICE_TYPE=cpu ctest --timeout 180 -j 8 --output-on-failure -E "`cat ./test_lists/cpu_pocl_failed_tests.txt`" | tee cpu_pocl_make_check_result.txt
echo "end cpu_pocl_failed_tests"

# Test Level Zero iGPU
echo "begin igpu_level0_failed_tests"
sudo /opt/ocl-icd/scripts/igpu_bind &> /dev/null
clinfo -l
CHIP_BE=level0 ctest --timeout 180 -j 1 --output-on-failure -R "hip_sycl_interop$"
CHIP_BE=level0 /home/pvelesko/CHIP-SPV/chip-spv/build/samples/hip_sycl_interop/hip_sycl_interop
CHIP_BE=level0 ctest --timeout 180 -j 1 --output-on-failure -E "`cat ./test_lists/igpu_level0_failed_tests.txt`" | tee igpu_level0_make_check_result.txt
CHIP_BE=level0 ctest --timeout 180 -j 1 --output-on-failure -R "hip_sycl_interop$"
CHIP_BE=level0 /home/pvelesko/CHIP-SPV/chip-spv/build/samples/hip_sycl_interop/hip_sycl_interop
sudo /opt/ocl-icd/scripts/igpu_unbind &> /dev/null
echo "end igpu_level0_failed_tests"

# Test Level Zero dGPU
echo "begin dgpu_level0_failed_tests"
sudo /opt/ocl-icd/scripts/dgpu_bind &> /dev/null
clinfo -l
CHIP_BE=level0 ctest --timeout 180 -j 1 --output-on-failure -R "hip_sycl_interop$"
CHIP_BE=level0 /home/pvelesko/CHIP-SPV/chip-spv/build/samples/hip_sycl_interop/hip_sycl_interop
CHIP_BE=level0 ctest --timeout 180 -j 1 --output-on-failure -E "`cat ./test_lists/dgpu_level0_failed_tests.txt`" | tee dgpu_level0_make_check_result.txt
CHIP_BE=level0 ctest --timeout 180 -j 1 --output-on-failure -R "hip_sycl_interop$"
CHIP_BE=level0 /home/pvelesko/CHIP-SPV/chip-spv/build/samples/hip_sycl_interop/hip_sycl_interop
sudo /opt/ocl-icd/scripts/dgpu_unbind &> /dev/null
echo "end dgpu_level0_failed_tests"

# Test OpenCL iGPU
echo "begin igpu_opencl_failed_tests"
module swap opencl opencl/intel-gpu
sudo /opt/ocl-icd/scripts/igpu_bind &> /dev/null
clinfo -l
CHIP_BE=opencl ctest --timeout 180 -j 8 --output-on-failure -E "`cat ./test_lists/igpu_opencl_failed_tests.txt`" | tee igpu_opencl_make_check_result.txt
sudo /opt/ocl-icd/scripts/igpu_unbind &> /dev/null
echo "end igpu_opencl_failed_tests"

# Test OpenCL dGPU
echo "begin dgpu_opencl_failed_tests"
sudo /opt/ocl-icd/scripts/dgpu_bind &> /dev/null
clinfo -l
CHIP_BE=opencl ctest --timeout 180 -j 8 --output-on-failure -E "`cat ./test_lists/dgpu_opencl_failed_tests.txt`" | tee dgpu_opencl_make_check_result.txt
sudo /opt/ocl-icd/scripts/dgpu_unbind &> /dev/null
echo "end dgpu_opencl_failed_tests"

# Test OpenCL CPU
echo "begin cpu_opencl_failed_tests"
module swap opencl opencl/intel-cpu
clinfo -l
CHIP_BE=opencl CHIP_DEVICE_TYPE=cpu ctest --timeout 180 -j 8 --output-on-failure -E "`cat ./test_lists/cpu_opencl_failed_tests.txt`" | tee cpu_opencl_make_check_result.txt
echo "end cpu_opencl_failed_tests"

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

if [[ `cat cpu_pocl_make_check_result.txt | grep "0 tests failed out of"` ]]
then
    echo "PoCL OpenCL PASS"
    CPU_PoCL=1
else
    echo "PoCL OpenCL FAIL"
    CPU_PoCL=0
fi

if [[ "$iGPU_OpenCL" -eq "1" && \
      "$dGPU_OpenCL" -eq "1" && \
      "$CPU_PoCL" -eq "1" && \
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