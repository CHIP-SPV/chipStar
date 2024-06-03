#!/bin/bash

set -ex

source /etc/profile.d/modules.sh &> /dev/null
module use ~/modulefiles

export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1
export CHIP_LOGLEVEL=err

module purge
module load llvm/18.0
module load HIP/chipStar/testing

cd /home/pvelesko/libCEED
git co HEAD -f
git pull
make clean

make FC= CC=hipcc CXX=hipcc BACKENDS="/gpu/hip/ref /gpu/hip/shared /gpu/hip/gen" -j $(nproc) test | tee libceed_output.txt
if grep -q "not ok" "libceed_output.txt"; then
  echo "FAIL"
  awk '/Test Summary Report/,EOF' "libceed_output.txt"
  exit 1
else
  echo "PASS"
  exit 0
fi
