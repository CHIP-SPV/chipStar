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
cd ../
sudo /opt/ocl-icd/scripts/igpu_unbind &> /dev/null

# dgpu
sudo /opt/ocl-icd/scripts/dgpu_bind &> /dev/null
./scripts/check_for_resolved_tests.py ./build dgpu opencl 1 100
./scripts/check_for_resolved_tests.py ./build dgpu level0 1 100
sudo /opt/ocl-icd/scripts/dgpu_unbind &> /dev/null

# igpu
sudo /opt/ocl-icd/scripts/igpu_bind &> /dev/null
./scripts/check_for_resolved_tests.py ./build idgpu opencl 1 100
./scripts/check_for_resolved_tests.py ./build igpu level0 1 100
sudo /opt/ocl-icd/scripts/igpu_unbind &> /dev/null

# cpu
./scripts/check_for_resolved_tests.py ./build cpu opencl 1 100
./scripts/check_for_resolved_tests.py ./build cpu pocl 1 100

echo "dgpu opencl"
cat ./build/dgpu_opencl_resolved_tests.txt
echo ""

echo "dgpu level0"
cat ./build/dgpu_level0_resolved_tests.txt
echo ""

echo "igpu opencl"
cat ./build/igpu_opencl_resolved_tests.txt
echo ""

echo "igpu level0"
cat ./build/igpu_level0_resolved_tests.txt
echo ""

echo "cpu opencl"
cat ./build/cpu_opencl_resolved_tests.txt
echo ""

echo "cpu pocl"
cat ./build/cpu_pocl_resolved_tests.txt
echo ""
