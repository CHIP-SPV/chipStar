#!/bin/bash

source /opt/intel/oneapi/setvars.sh intel64 &> /dev/null
source /etc/profile.d/modules.sh
export MODULEPATH=$MODULEPATH:/home/pvelesko/modulefiles:/opt/intel/oneapi/modulefiles
export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1

module load mkl opencl/intel-gpu

# Ensure that only igpu is active for build/test discovery and OpenCL is used
sudo /opt/ocl-icd/scripts/dgpu_unbind &> /dev/null
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
