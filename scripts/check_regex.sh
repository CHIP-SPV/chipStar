#!/bin/bash

set -e

# Check if there are at least three arguments
if [ $# -lt 1 ]; then
  echo "Usage: $0 <rgx> [--num-tries=<value>] [--num-threads=<value>] [modules...]"
  exit 1
fi


# Save the first argument to variable rgx
rgx="$1"

# Set default values for num_threads and num_tries
num_threads=1
num_tries=1

# Check if the arguments include --num-threads and/or --num-tries
for arg in "$@"
do
  case $arg in
    --num-threads=*)
      num_threads="${arg#*=}"
      shift
      ;;
    --num-tries=*)
      num_tries="${arg#*=}"
      shift
      ;;
    *)
      ;;
  esac
done

# Check if the second and third arguments are integers
if ! [[ "$num_threads" =~ ^[0-9]+$ ]] || ! [[ "$num_tries" =~ ^[0-9]+$ ]]; then
  echo "Error: --num-threads and --num-tries arguments must be integers."
  exit 1
fi

# Check if the number of threads and tries are greater than 0
if [ "$num_threads" -lt 1 ] || [ "$num_tries" -lt 1 ]; then
  echo "Error: --num-threads and --num-tries arguments must be greater than 0."
  exit 1
fi

source /etc/profile.d/modules.sh &> /dev/null
export MODULEPATH=$MODULEPATH:/home/pvelesko/modulefiles:/opt/intel/oneapi/modulefiles
export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1
export CHIP_LOGLEVEL=err

compiler="clang/clang17-spirv-omp"

igpu_level0="level-zero/igpu"
dgpu_level0="level-zero/igpu"
igpu_opencl="opencl/igpu"
dgpu_opencl="opencl/dgpu"
cpu_opencl="opencl/cpu"
cpu_pocl="opencl/pocl"

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
  ctest -R "$rgx" -j "$num_threads" --repeat until-fail:"$num_tries" --timeout 180 --output-on-failure | tee "$TMP_FILE"
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