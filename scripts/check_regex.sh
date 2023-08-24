#!/bin/bash

# Check if there are at least three arguments
if [ $# -lt 3 ]; then
  echo "Usage: $0 <rgx> <NUM_TRIES> <NUM_THREADS> [modules...]"
  exit 1
fi

# Check if 2nd and 3rd arguments are integers
if ! [[ "$2" =~ ^[0-9]+$ ]] || ! [[ "$3" =~ ^[0-9]+$ ]]; then
  echo "2nd and 3rd arguments must be integers"
  exit 1
fi

source /etc/profile.d/modules.sh &> /dev/null
export MODULEPATH=$MODULEPATH:/home/pvelesko/modulefiles:/opt/intel/oneapi/modulefiles
export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1
export CHIP_LOGLEVEL=err

# Save the first argument to variable rgx
rgx="$1"
# Save the second argument to variable NUM_TRIES
NUM_TRIES="$2"
# Save the third argument to variable NUM_THREADS
NUM_THREADS="$3"

# Shift to get rid of the first three arguments
shift 3

compiler="clang/clang16-spirv-omp"

igpu_level0="levelzero/igpu"
dgpu_level0="levelzero/igpu"
igpu_opencl="opencl/intel-igpu"
dgpu_opencl="opencl/intel-dgpu"
cpu_opencl="opencl/intel-cpu"
cpu_pocl="opencl/pocl-cpu-llvm-16"

module purge

FINAL_FILE=check_regex_${rgx}.txt

rm -f $FINAL_FILE

function run_tests {
  local PLATFORM="$1"
  local SAFE_PLATFORM=$(echo "$PLATFORM" | sed 's/[^a-zA-Z0-9]/_/g')
  local SAFE_RGX=$(echo "$rgx" | sed 's/[^a-zA-Z0-9]/_/g')
  TMP_FILE="check_regex_${SAFE_RGX}_${SAFE_PLATFORM}.tmp"
  module load "$compiler" $PLATFORM mkl
  echo "Testing $PLATFORM"
  ctest -R "$rgx" -j "$NUM_THREADS" --repeat until-fail:"$NUM_TRIES" --timeout 180 --output-on-failure | tee "$TMP_FILE"
  echo "begin ${PLATFORM}" >> $FINAL_FILE
  awk '/tests passed, / {flag=1} flag' "$TMP_FILE" >> $FINAL_FILE
  echo "end ${PLATFORM}" >> $FINAL_FILE
  echo "" >> $FINAL_FILE
  rm "$TMP_FILE"
  module purge
}



run_tests "$igpu_level0"
run_tests "$dgpu_level0"
run_tests "$igpu_opencl"
run_tests "$dgpu_opencl"
run_tests "$cpu_opencl"
run_tests "$cpu_pocl"

echo ""
echo Done!
echo ""

cat "$FINAL_FILE"