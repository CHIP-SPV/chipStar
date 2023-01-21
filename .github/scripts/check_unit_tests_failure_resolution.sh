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
CHIP_BE=opencl CHIP_DEVICE_TYPE=gpu CHIP_PLATFORM=4 CHIP_DEVICE=0 ctest --timeout 180 -j 8  -R "`cat ./test_lists/igpu_opencl_failed_tests.txt`" | tee igpu_opencl_make_check_result.txt
echo "end igpu_opencl_failed_tests"
# Test OpenCL dGPU
echo "begin dgpu_opencl_failed_tests"
CHIP_BE=opencl CHIP_DEVICE_TYPE=gpu CHIP_PLATFORM=3 CHIP_DEVICE=0 ctest --timeout 180 -j 8  -R "`cat ./test_lists/dgpu_opencl_failed_tests.txt`" | tee dgpu_opencl_make_check_result.txt
echo "end dgpu_opencl_failed_tests"
# Test OpenCL CPU
echo "begin cpu_opencl_failed_tests"
CHIP_BE=opencl CHIP_DEVICE_TYPE=cpu CHIP_PLATFORM=1 CHIP_DEVICE=0 ctest --timeout 180 -j 8  -R "`cat ./test_lists/cpu_opencl_failed_tests.txt`" | tee cpu_opencl_make_check_result.txt
echo "end cpu_opencl_failed_tests"
# Test Level Zero iGPU
echo "begin igpu_level0_failed_tests"
CHIP_BE=level0 CHIP_DEVICE=1 ctest --timeout 180 -j 1  -R "`cat ./test_lists/igpu_level0_failed_tests.txt`" | tee igpu_level0_make_check_result.txt
echo "end igpu_level0_failed_tests"
# Test Level Zero dGPU
echo "begin dgpu_level0_failed_tests"
CHIP_BE=level0 CHIP_DEVICE=0 ctest --timeout 180 -j 1  -R "`cat ./test_lists/dgpu_level0_failed_tests.txt`" | tee dgpu_level0_make_check_result.txt
echo "end dgpu_level0_failed_tests"

cat igpu_opencl_make_check_result.txt | grep Passed | awk  '{ print $4 }' > igpu_opencl_maybe_passed.txt
cat dgpu_opencl_make_check_result.txt | grep Passed | awk  '{ print $4 }' > dgpu_opencl_maybe_passed.txt
cat  cpu_opencl_make_check_result.txt | grep Passed | awk  '{ print $4 }' >  cpu_opencl_maybe_passed.txt
cat igpu_level0_make_check_result.txt | grep Passed | awk  '{ print $4 }' > igpu_level0_maybe_passed.txt
cat dgpu_level0_make_check_result.txt | grep Passed | awk  '{ print $4 }' > dgpu_level0_maybe_passed.txt

cat igpu_opencl_make_check_result.txt | grep Passed | awk  -vORS="$|" '{ print $4 }' > igpu_opencl_passed_result.txt
cat dgpu_opencl_make_check_result.txt | grep Passed | awk  -vORS="$|" '{ print $4 }' > dgpu_opencl_passed_result.txt
cat  cpu_opencl_make_check_result.txt | grep Passed | awk  -vORS="$|" '{ print $4 }' >  cpu_opencl_passed_result.txt
cat igpu_level0_make_check_result.txt | grep Passed | awk  -vORS="$|" '{ print $4 }' > igpu_level0_passed_result.txt
cat dgpu_level0_make_check_result.txt | grep Passed | awk  -vORS="$|" '{ print $4 }' > dgpu_level0_passed_result.txt

echo "\"" > igpu_opencl_passed.txt
cat igpu_opencl_passed_result.txt >> igpu_opencl_passed.txt
rm igpu_opencl_passed_result.txt
echo "\"" >> igpu_opencl_passed.txt

echo "\"" > dgpu_opencl_passed.txt
cat dgpu_opencl_passed_result.txt >> dgpu_opencl_passed.txt
rm dgpu_opencl_passed_result.txt
echo "\"" >> dgpu_opencl_passed.txt

echo "\"" > cpu_opencl_passed.txt
cat cpu_opencl_passed_result.txt >> cpu_opencl_passed.txt
rm cpu_opencl_passed_result.txt
echo "\"" >> cpu_opencl_passed.txt

echo "\"" > igpu_level0_passed.txt
cat igpu_level0_passed_result.txt >> igpu_level0_passed.txt
rm igpu_level0_passed_result.txt
echo "\"" >> igpu_level0_passed.txt

echo "\"" > dgpu_level0_passed.txt
cat dgpu_level0_passed_result.txt >> dgpu_level0_passed.txt
rm dgpu_level0_passed_result.txt
echo "\"" >> dgpu_level0_passed.txt

### Verify that each test that failed, passes multiple times
# Test OpenCL iGPU
echo "begin igpu_opencl_passed_tests"
CHIP_BE=opencl CHIP_DEVICE_TYPE=gpu CHIP_PLATFORM=4 CHIP_DEVICE=0 ctest --timeout 180 -j 8 --repeat until-fail:100 -R "`cat ./igpu_opencl_passed.txt`" | tee igpu_opencl_resolved_failures.txt
echo "end igpu_opencl_passed_tests"
# Test OpenCL dGPU
echo "begin dgpu_opencl_passed_tests"
CHIP_BE=opencl CHIP_DEVICE_TYPE=gpu CHIP_PLATFORM=3 CHIP_DEVICE=0 ctest --timeout 180 -j 8 --repeat until-fail:100 -R "`cat ./dgpu_opencl_passed.txt`" | tee dgpu_opencl_resolved_failures.txt
echo "end dgpu_opencl_passed_tests"
# Test OpenCL CPU
echo "begin cpu_opencl_passed_tests"
CHIP_BE=opencl CHIP_DEVICE_TYPE=cpu CHIP_PLATFORM=1 CHIP_DEVICE=0 ctest --timeout 180 -j 8 --repeat until-fail:100 -R "`cat ./cpu_opencl_passed.txt`" | tee cpu_opencl_resolved_failures.txt
echo "end cpu_opencl_passed_tests"
# Test Level Zero iGPU
echo "begin igpu_level0_passed_tests"
CHIP_BE=level0 CHIP_DEVICE=1 ctest --timeout 180 -j 1 --repeat until-fail:100 -R "`cat ./igpu_level0_passed.txt`" | tee igpu_level0_resolved_failures.txt
echo "end igpu_level0_passed_tests"
# Test Level Zero dGPU
echo "begin dgpu_level0_passed_tests"
CHIP_BE=level0 CHIP_DEVICE=0 ctest --timeout 180 -j 1 --repeat until-fail:100 -R "`cat ./dgpu_level0_passed.txt`" | tee dgpu_level0_resolved_failures.txt
echo "end dgpu_level0_passed_tests"
