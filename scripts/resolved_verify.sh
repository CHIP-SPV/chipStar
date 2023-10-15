#!/bin/bash

set -eu

source /etc/profile.d/modules.sh &> /dev/null
export MODULEPATH=$MODULEPATH:/home/pvelesko/modulefiles:/opt/intel/oneapi/modulefiles:/opt/modulefiles

module load intel/compute-runtime intel/opencl
module load level-zero/dgpu

./scripts/check_for_resolved_tests.py ./build dgpu level0-reg 4 1000 verify &
./scripts/check_for_resolved_tests.py ./build dgpu level0-imm 4 1000 verify &
module unload level-zero/dpgu

module load level-zero/igpu
./scripts/check_for_resolved_tests.py ./build igpu level0-reg 4 1000 verify &
module unload level-zero/igpu

module load opencl/dgpu
./scripts/check_for_resolved_tests.py ./build dgpu opencl 4 1000 verify &
module unload opencl/dgpu

module load opencl/igpu
./scripts/check_for_resolved_tests.py ./build igpu opencl 4 1000 verify &
module unload opencl/igpu


module load opencl/pocl
./scripts/check_for_resolved_tests.py ./build cpu pocl 4 1000 verify &
module unload opencl/pocl

module load opencl/cpu
./scripts/check_for_resolved_tests.py ./build cpu opencl 4 1000 verify &
module unload opencl/cpu