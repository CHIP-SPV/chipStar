#!/bin/bash

set -e

export CHIP_MODULE_CACHE_DIR=""
host=`hostname`
echo "Running on ${host}"
# If not on Salami read the file /opt/actions-runner/num-threads.txt and set the number of threads to the value in the file
# if the file does not exist, set the number of threads to 24
if [ -f "/opt/actions-runner/num-threads.txt" ]; then
  num_threads=$(cat /opt/actions-runner/num-threads.txt)
else
  num_threads=$(nproc)
fi

num_tries=1
# deafult timeout is 30 minutes
timeout=1800
build_only=false

# Check if at least one argument is provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <debug|release> <llvm-16|llvm-17|llvm-18|llvm-19> [--skip-build] [--build-only] [--num-tries=$num_tries] [--num-threads=$num_threads] [--timeout=$timeout]"
  exit 1
fi

# Check if the first argument is either "debug" or "release"
if [ "$1" != "debug" ] && [ "$1" != "release" ]; then
  echo "Error: Invalid argument. Must be either 'debug' or 'release'."
  exit 1
fi

# Set the build type based on the argument
build_type=$(echo "$1" | tr '[:lower:]' '[:upper:]')

# Check if the second argument starts with "llvm-" and is followed by a valid version number
if [[ ! "$2" =~ ^llvm-(1[6-9]|[2-9][0-9])$ ]]; then
  echo "Error: Invalid LLVM version. Must be llvm-16, llvm-17, llvm-18, or higher."
  exit 1
fi

if [ "$host" = "salami" ]; then
  LLVM="llvm-${2#llvm-}"
  CLANG="llvm/${2#llvm-}.0-exts-only"
else
  LLVM="llvm-${2#llvm-}"
  CLANG="llvm/${2#llvm-}"
fi

shift
shift

for arg in "$@"
do
  case $arg in
    --num-threads=*)
      num_threads="${arg#*=}"
      shift
      ;;
    --num-threads)
      shift
      num_threads="$1"
      shift
      ;;
    --num-tries=*)
      num_tries="${arg#*=}"
      shift
      ;;
    --num-tries)
      shift
      num_tries="$1"
      shift
      ;;
    --skip-build)
      skip_build=true
      shift
      ;;
    --build-only)
      build_only=true
      shift
      ;;
    --timeout=*)
      timeout="${arg#*=}"
      shift
      ;;
    --timeout)
      shift
      timeout="$1"
      shift
      ;;
    *)
      ;;
  esac
done


# Print out the arguments
echo "build_type  = ${build_type}"
echo "LLVM        = ${LLVM}"
echo "CLANG       = ${CLANG}"
echo "num_tries   = ${num_tries}"
echo "num_threads = ${num_threads}"
echo "skip_build  = ${skip_build}"
echo "build_only  = ${build_only}"
echo "timeout     = ${timeout}"

# source /opt/intel/oneapi/setvars.sh intel64 &> /dev/null
source /etc/profile.d/modules.sh &> /dev/null
export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1
export CHIP_LOGLEVEL=err
export POCL_KERNEL_CACHE=0

# set event timeout to be 10 less than timeout
export CHIP_L0_EVENT_TIMEOUT=$(($timeout - 10))

# Use OpenCL for building/test discovery to prevent Level Zero from being used in multi-thread/multi-process environment
if [ "$host" = "salami" ]; then
  module use  /home/kristian/apps/modulefiles
  module load $CLANG
else
  module use ~/modulefiles
  module load oneapi/2024.1.0 $CLANG opencl/dgpu 
fi

output=$(clinfo -l 2>&1 | grep "Platform #0")
echo $output
if [ $? -ne 0 ]; then
    echo "clinfo failed to execute."
    exit 1
fi

# check if the output is empty
if [ -z "$output" ]; then
    echo "No OpenCL devices detected."
    exit 1
fi

if [ $skip_build ]; then
  echo "Skipping build step"
  cd build
else
  if [ "$host" = "salami" ]; then
    CHIP_OPTIONS="-DCHIP_MALI_GPU_WORKAROUNDS=ON -DCHIP_SKIP_TESTS_WITH_DOUBLES=ON -DCHIP_BUILD_SAMPLES=ON"
  else
    CHIP_OPTIONS="-DCHIP_BUILD_HIPBLAS=ON -DCHIP_BUILD_HIPFFT=ON -DCHIP_BUILD_HIPSOLVER=ON -DCHIP_BUILD_SAMPLES=ON"
  fi
  # Build the project
  echo "Building project..."
  rm -rf HIPCC
  rm -rf HIP
  rm -rf bitcode/ROCm-Device-Libs
  rm -rf hip-tests
  rm -rf hip-testsuite

  git submodule update --init
  rm -rf build
  rm -rf *_result.txt
  mkdir build
  cd build

  echo "building with $CLANG"
  cmake ../ -DCMAKE_BUILD_TYPE="$build_type"  ${CHIP_OPTIONS}
  make all build_tests install -j $(nproc) #&> /dev/null
  echo "chipStar build complete." 
fi

if [ "$build_only" = true ]; then
  echo "Build-only mode. Exiting."
  exit 0
fi

module unload opencl/dgpu

# Function to run tests
run_tests() {
    local device=$1
    local backend=$2
    echo "begin ${device}_${backend}_failed_tests"
    ../scripts/check.py ./ $device $backend --num-threads=${num_threads} --timeout=$timeout --num-tries=$num_tries --modules=on | tee ${device}_${backend}_make_check_result.txt
    echo "end ${device}_${backend}_failed_tests"
}

function check_tests {
  file="$1"
  if ! grep -q "The following tests FAILED" "$file"; then
    return 0
  else
    echo "Checking $file"
    grep -E "The following tests FAILED:" -A 1000 "$file" | sed '/^$/q' | tail -n +2
    return 1
  fi
}

set +e # disable exit on error

# Run tests for different configurations
run_tests igpu opencl
if [ "$host" = "salami" ]; then
  check_tests igpu_opencl_make_check_result.txt
  igpu_opencl_exit_code=$?
  exit $igpu_opencl_exit_code
fi
run_tests igpu level0
run_tests dgpu level0
run_tests dgpu opencl
run_tests cpu opencl

check_tests igpu_opencl_make_check_result.txt
igpu_opencl_exit_code=$?
check_tests cpu_opencl_make_check_result.txt
cpu_opencl_exit_code=$?
check_tests dgpu_opencl_make_check_result.txt
dgpu_opencl_exit_code=$?
check_tests dgpu_level0_make_check_result.txt
dgpu_level0_exit_code=$?
check_tests igpu_level0_make_check_result.txt
igpu_level0_exit_code=$?

exit $((igpu_opencl_exit_code || dgpu_opencl_exit_code || igpu_level0_exit_code || dgpu_level0_exit_code || cpu_opencl_exit_code))